# Copyright 2023 Sony Group Corporation.

import torch
import torch.nn
import numpy as np
import cv2
import json

from net.net_seld import create_net_seld
from seld_trainer import MSELoss_ADPIT
from dcase2022_task3_seld_metrics.SELD_evaluation_metrics import distance_between_spherical_coordinates_rad

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)  # temporary for pandas
import pandas as pd


class SELDClassifier(object):
    def __init__(self, args, object_detection):
        self._args = args

        self._device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self._object_detection = object_detection

        self._net = create_net_seld(self._args)
        self._net.to(self._device)
        self._net.eval()
        checkpoint = torch.load(self._args.eval_model, map_location=lambda storage, loc: storage)
        self._net.load_state_dict(checkpoint['model_state_dict'])

        self._criterion = MSELoss_ADPIT()

    def set_input(self, spec_pad, label_pad, video):
        self._spec_pad = spec_pad
        self._label_pad = label_pad
        self._cap = video

    def receive_input(self, time_array):
        fs = self._args.sampling_frequency
        self._frame_per_sec = round(fs / self._args.stft_hop_size)
        self._frame_length = round(self._args.train_wav_length * fs / self._args.stft_hop_size) + 1
        features = np.zeros(tuple([self._args.batch_size]) + (self._spec_pad[:, :, :self._frame_length]).shape)
        labels = np.zeros(tuple([self._args.batch_size]) + (self._label_pad[:, :, :, :self._frame_length]).shape)
        videos = np.zeros(tuple([self._args.batch_size]) + tuple(self._object_detection.get_dist_shape()))

        for index, time in enumerate(time_array):
            frame_idx = int(time * self._frame_per_sec)
            features[index] = self._spec_pad[:, :, frame_idx: frame_idx + self._frame_length]
            labels[index] = self._label_pad[:, :, :, frame_idx: frame_idx + self._frame_length]

            self._cap.set(cv2.CAP_PROP_POS_FRAMES, int(self._cap.get(cv2.CAP_PROP_FPS) * time))
            _, frame = self._cap.read()
            frame_rgb_in = frame[:, :, [2, 1, 0]]  # BGR -> RGB
            box_in = self._object_detection.img2box(frame_rgb_in)
            frame_out = self._object_detection.box2dist(box_in)
            videos[index] = frame_out

        self._input_a = torch.tensor(features, dtype=torch.float).to(self._device)
        self._input_v = torch.tensor(videos, dtype=torch.float).to(self._device)
        self._label = torch.tensor(labels, dtype=torch.float).to(self._device)

    def calc_output(self):
        self._output = self._net(self._input_a, self._input_v)

    def get_output(self):
        hop_frame = round(self._args.eval_wav_hop_length * self._frame_per_sec)
        cut_frame = int(np.floor((self._frame_length - hop_frame) / 2))
        output = self._output.cpu().detach().numpy()
        self._output = 0  # for memory release
        return output[:, :, :, :, cut_frame: cut_frame + hop_frame]  # only use output from cut [frame] to cut + hop [frame]

    def get_loss(self):
        self._loss = self._criterion(self._output, self._label)
        loss = self._loss.cpu().detach().numpy()
        self._loss = 0  # for memory release
        return loss


class SELDDetector(object):
    def __init__(self, args):
        self._args = args
        with open(args.threshold_config, 'r') as f:
            threshold_config = json.load(f)
        self._thresh_bin = threshold_config['threshold_presence']
        self._thresh_dist = threshold_config['threshold_unification']

        fs = self._args.sampling_frequency
        self._frame_per_sec = round(fs / self._args.stft_hop_size)
        self._hop_frame = round(self._args.eval_wav_hop_length * self._frame_per_sec)

    def set_duration(self, duration):
        eval_wav_hop_length = self._args.eval_wav_hop_length
        if (duration % eval_wav_hop_length == 0) or (np.abs((duration % eval_wav_hop_length) - eval_wav_hop_length) < 1e-10):
            self._time_array = np.arange(0, duration + eval_wav_hop_length, eval_wav_hop_length)
        else:
            self._time_array = np.arange(0, duration, eval_wav_hop_length)

        self._df = pd.DataFrame()
        self._minibatch_result = np.zeros((
            len(self._time_array) + self._args.batch_size,
            3,
            3,
            self._args.class_num,
            self._hop_frame))
        self._raw_output_array = np.zeros((
            3,
            3,
            self._args.class_num,
            len(self._time_array) * self._hop_frame))

    def get_time_array(self):
        return self._time_array

    def set_minibatch_result(self, index, result):
        self._minibatch_result[
            index * self._args.batch_size: (index + 1) * self._args.batch_size
        ] = result

    def minibatch_result2raw_output_array(self):
        array_len = (self._minibatch_result.shape[0]) * self._hop_frame
        result_array = np.zeros((3, 3, self._args.class_num, array_len))
        for index, each_result in enumerate(self._minibatch_result):
            result_array[
                :, :, :, index * self._hop_frame: (index + 1) * self._hop_frame
            ] = each_result
        self._raw_output_array = result_array[:, :, :, : len(self._time_array) * self._hop_frame]

    def detect(self, index, time):
        for event_class in range(self._args.class_num):
            x0 = self._raw_output_array[0, 0, event_class, index * self._hop_frame: (index + 1) * self._hop_frame]
            y0 = self._raw_output_array[0, 1, event_class, index * self._hop_frame: (index + 1) * self._hop_frame]
            z0 = self._raw_output_array[0, 2, event_class, index * self._hop_frame: (index + 1) * self._hop_frame]
            x1 = self._raw_output_array[1, 0, event_class, index * self._hop_frame: (index + 1) * self._hop_frame]
            y1 = self._raw_output_array[1, 1, event_class, index * self._hop_frame: (index + 1) * self._hop_frame]
            z1 = self._raw_output_array[1, 2, event_class, index * self._hop_frame: (index + 1) * self._hop_frame]
            x2 = self._raw_output_array[2, 0, event_class, index * self._hop_frame: (index + 1) * self._hop_frame]
            y2 = self._raw_output_array[2, 1, event_class, index * self._hop_frame: (index + 1) * self._hop_frame]
            z2 = self._raw_output_array[2, 2, event_class, index * self._hop_frame: (index + 1) * self._hop_frame]
            self._each_detect(time, event_class, x0, y0, z0, x1, y1, z1, x2, y2, z2)

    def _each_detect(self, time, event_class, x0, y0, z0, x1, y1, z1, x2, y2, z2):
        azi0, ele0, bin0 = self._xyz2azi_ele_bin(x0, y0, z0)
        azi1, ele1, bin1 = self._xyz2azi_ele_bin(x1, y1, z1)
        azi2, ele2, bin2 = self._xyz2azi_ele_bin(x2, y2, z2)

        frame_per_sec4csv = 10  # for csv setting
        hop_frame4csv = int(self._hop_frame / (self._frame_per_sec / frame_per_sec4csv))  # e.g., 12 [frame in csv]
        for csv_idx, frame in enumerate(range(int(time * frame_per_sec4csv), int(time * frame_per_sec4csv) + hop_frame4csv)):
            csv2net = int(self._frame_per_sec / frame_per_sec4csv)  # e.g., 100 [frame for net] / 10 [frame for csv]
            net_idx_start = csv_idx * csv2net
            net_idx_end = (csv_idx + 1) * csv2net
            azi_mean0, ele_mean0, bin_mean0 = self._azi_ele_bin2mean(azi0, ele0, bin0, net_idx_start, net_idx_end, event_class)
            azi_mean1, ele_mean1, bin_mean1 = self._azi_ele_bin2mean(azi1, ele1, bin1, net_idx_start, net_idx_end, event_class)
            azi_mean2, ele_mean2, bin_mean2 = self._azi_ele_bin2mean(azi2, ele2, bin2, net_idx_start, net_idx_end, event_class)

            # x is similar to y?
            flag_0sim1 = self._similar_location(azi_mean0, ele_mean0, azi_mean1, ele_mean1)
            flag_1sim2 = self._similar_location(azi_mean1, ele_mean1, azi_mean2, ele_mean2)
            flag_2sim0 = self._similar_location(azi_mean2, ele_mean2, azi_mean0, ele_mean0)

            # unify or not according to flag
            if flag_0sim1 + flag_1sim2 + flag_2sim0 == 0:
                if bin_mean0 > self._thresh_bin[event_class]:
                    self._df = self._df.append(pd.DataFrame([(frame, event_class, azi_mean0 / np.pi * 180, ele_mean0 / np.pi * 180)]))
                if bin_mean1 > self._thresh_bin[event_class]:
                    self._df = self._df.append(pd.DataFrame([(frame, event_class, azi_mean1 / np.pi * 180, ele_mean1 / np.pi * 180)]))
                if bin_mean2 > self._thresh_bin[event_class]:
                    self._df = self._df.append(pd.DataFrame([(frame, event_class, azi_mean2 / np.pi * 180, ele_mean2 / np.pi * 180)]))
            elif flag_0sim1 + flag_1sim2 + flag_2sim0 == 1:
                if flag_0sim1:
                    if bin_mean2 > self._thresh_bin[event_class]:
                        self._df = self._df.append(pd.DataFrame([(frame, event_class, azi_mean2 / np.pi * 180, ele_mean2 / np.pi * 180)]))
                    azi_mean_unified_12 = (azi_mean0 * bin_mean0 + azi_mean1 * bin_mean1) / (bin_mean0 + bin_mean1)
                    ele_mean_unified_12 = (ele_mean0 * bin_mean0 + ele_mean1 * bin_mean1) / (bin_mean0 + bin_mean1)
                    self._df = self._df.append(pd.DataFrame([(frame, event_class, azi_mean_unified_12 / np.pi * 180, ele_mean_unified_12 / np.pi * 180)]))
                elif flag_1sim2:
                    if bin_mean0 > self._thresh_bin[event_class]:
                        self._df = self._df.append(pd.DataFrame([(frame, event_class, azi_mean0 / np.pi * 180, ele_mean0 / np.pi * 180)]))
                    azi_mean_unified_23 = (azi_mean1 * bin_mean1 + azi_mean2 * bin_mean2) / (bin_mean1 + bin_mean2)
                    ele_mean_unified_23 = (ele_mean1 * bin_mean1 + ele_mean2 * bin_mean2) / (bin_mean1 + bin_mean2)
                    self._df = self._df.append(pd.DataFrame([(frame, event_class, azi_mean_unified_23 / np.pi * 180, ele_mean_unified_23 / np.pi * 180)]))
                elif flag_2sim0:
                    if bin_mean1 > self._thresh_bin[event_class]:
                        self._df = self._df.append(pd.DataFrame([(frame, event_class, azi_mean1 / np.pi * 180, ele_mean1 / np.pi * 180)]))
                    azi_mean_unified_31 = (azi_mean2 * bin_mean2 + azi_mean0 * bin_mean0) / (bin_mean2 + bin_mean0)
                    ele_mean_unified_31 = (ele_mean2 * bin_mean2 + ele_mean0 * bin_mean0) / (bin_mean2 + bin_mean0)
                    self._df = self._df.append(pd.DataFrame([(frame, event_class, azi_mean_unified_31 / np.pi * 180, ele_mean_unified_31 / np.pi * 180)]))
            elif flag_0sim1 + flag_1sim2 + flag_2sim0 >= 2:
                azi_mean_unified_123 = (azi_mean0 * bin_mean0 + azi_mean1 * bin_mean1 + azi_mean2 * bin_mean2) / (bin_mean0 + bin_mean1 + bin_mean2)
                ele_mean_unified_123 = (ele_mean0 * bin_mean0 + ele_mean1 * bin_mean1 + ele_mean2 * bin_mean2) / (bin_mean0 + bin_mean1 + bin_mean2)
                self._df = self._df.append(pd.DataFrame([(frame, event_class, azi_mean_unified_123 / np.pi * 180, ele_mean_unified_123 / np.pi * 180)]))

    def _xyz2azi_ele_bin(self, x, y, z):
        azi = np.arctan2(y, x)
        ele = np.arctan2(z, np.sqrt(x**2 + y**2))
        bin = np.sqrt(x**2 + y**2 + z**2)
        bin[bin > 1] = 1
        return azi, ele, bin

    def _azi_ele_bin2mean(self, azi, ele, bin, idx_start, idx_end, event_class):
        bin_mean = np.mean(bin[idx_start: idx_end])
        azi_mean, ele_mean = None, None
        if bin_mean > self._thresh_bin[event_class]:
            azi_mean = np.sum(bin[idx_start: idx_end] * azi[idx_start: idx_end]) / np.sum(bin[idx_start: idx_end])
            ele_mean = np.sum(bin[idx_start: idx_end] * ele[idx_start: idx_end]) / np.sum(bin[idx_start: idx_end])
        return azi_mean, ele_mean, bin_mean

    def _similar_location(self, azi0, ele0, azi1, ele1):
        if (azi0 is not None) and (azi1 is not None):
            if distance_between_spherical_coordinates_rad(azi0, ele0, azi1, ele1) < self._thresh_dist:
                return 1
            else:
                return 0
        else:
            return 0

    def save_df(self, pred_path):
        if not self._df.empty:
            self._df = self._df.sort_values(0)
        self._df.to_csv(pred_path, sep=',', index=False, header=False)
