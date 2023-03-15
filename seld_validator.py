# Copyright 2023 Sony Group Corporation.

import cv2
import os
import tqdm
import pandas as pd

from wav_convertor import WavConvertor
from seld_predictor import SELDClassifier, SELDDetector
from seld_eval_dcase2022 import all_seld_eval


class SELDValidator(object):
    def __init__(self, args, object_detection, monitor_path):
        self._args = args
        self._object_detection = object_detection
        self._monitor_path = monitor_path
        if self._args.val:
            self._tag = 'TMP4VAL'
        elif self._args.eval:
            self._tag = '{}_{}'.format(
                os.path.splitext(os.path.basename(self._args.eval_wav_txt))[0],
                os.path.splitext(os.path.basename(self._args.eval_model))[0][-7:]  # iteration
            )
        self._pred_directory = os.path.join(monitor_path, 'pred_{}'.format(self._tag))
        os.makedirs(self._pred_directory, exist_ok=True)

        self._wav_convertor = WavConvertor(self._args)

        # wav on RAM
        if self._args.val:
            self._wav_path_list = pd.read_table(self._args.val_wav_txt, header=None).values.tolist()
        elif self._args.eval:
            self._wav_path_list = pd.read_table(self._args.eval_wav_txt, header=None).values.tolist()
        self._wav_dict = {}
        self._duration_dict = {}
        self._label_dict = {}
        for row in tqdm.tqdm(self._wav_path_list, desc='[Val initial setup]'):
            wav_path = row[0]
            wav_pad, duration = self._wav_convertor.wav_path2wav(wav_path)
            label_pad = self._wav_convertor.wav_path2label(wav_path, duration)
            self._wav_dict[wav_path] = wav_pad
            self._duration_dict[wav_path] = duration
            self._label_dict[wav_path] = label_pad

    def validation(self, model_path):
        self._args.eval_model = model_path  # temporary replace args for validation
        self._seld_classifier = SELDClassifier(self._args, self._object_detection)
        self._seld_detector = SELDDetector(self._args)
        val_loss = 0
        for row in tqdm.tqdm(self._wav_path_list, desc='[Val]'):
            wav_path = row[0]
            wav_loss = self._pred_wav(wav_path)
            val_loss += wav_loss
        val_loss = val_loss / len(self._wav_path_list)

        if self._args.val:
            all_test_metric = all_seld_eval(self._args, pred_directory=self._pred_directory)
        elif self._args.eval:
            result_path = os.path.join(self._monitor_path, 'result_{}.txt'.format(self._tag))
            all_test_metric = all_seld_eval(self._args, pred_directory=self._pred_directory,
                                            result_path=result_path)

        return all_test_metric, val_loss

    def _pred_wav(self, wav_path):
        # input setup
        spec_pad = self._wav_convertor.wav2spec(self._wav_dict[wav_path])
        duration = self._duration_dict[wav_path]
        label_pad = self._label_dict[wav_path]
        mp4_path = wav_path.replace('mic', 'video_360x180').replace('foa', 'video_360x180').replace('.wav', '.mp4')
        cv2_video = cv2.VideoCapture(mp4_path)
        frame_count = cv2_video.get(cv2.CAP_PROP_FRAME_COUNT)
        fps = cv2_video.get(cv2.CAP_PROP_FPS)
        if duration > frame_count / fps:
            duration = frame_count / fps  # update for time_array

        # classifier and detector setup
        self._seld_classifier.set_input(spec_pad, label_pad, cv2_video)
        self._seld_detector.set_duration(duration)
        time_array = self._seld_detector.get_time_array()

        # minibatch-like processing for classifier
        wav_loss = 0
        for index, time in enumerate(time_array[::self._args.batch_size]):
            self._seld_classifier.receive_input(
                time_array[index * self._args.batch_size: (index + 1) * self._args.batch_size])
            self._seld_classifier.calc_output()
            wav_loss += self._seld_classifier.get_loss()
            self._seld_detector.set_minibatch_result(
                index=index,
                result=self._seld_classifier.get_output()
            )
        self._seld_detector.minibatch_result2raw_output_array()
        wav_loss = wav_loss / len(time_array[::self._args.batch_size])

        # online-like processing for detector
        for index, time in enumerate(time_array):
            self._seld_detector.detect(index=index, time=time)
        pred_path = os.path.join(
            self._pred_directory,
            '{}.csv'.format(os.path.splitext(os.path.basename(wav_path))[0])
        )
        self._seld_detector.save_df(pred_path)

        return wav_loss
