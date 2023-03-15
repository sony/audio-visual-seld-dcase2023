# Copyright 2023 Sony Group Corporation.

import numpy as np
import random
import math


def select_time(train_wav_length, wav, fs):
    center = random.randrange(round(0        + train_wav_length / 2 * fs),
                              round(len(wav) - train_wav_length / 2 * fs))
    start = center - round(train_wav_length / 2 * fs)

    return start


def add_label_each_frame(label, time_array4frame_event, start_frame):
    event_class = time_array4frame_event[1]

    azi_rad = time_array4frame_event[3] / 180 * np.pi
    ele_rad = time_array4frame_event[4] / 180 * np.pi
    x_axis = 1 * np.cos(ele_rad) * np.cos(azi_rad)
    y_axis = 1 * np.cos(ele_rad) * np.sin(azi_rad)
    z_axis = 1 * np.sin(ele_rad)

    label[0, event_class, start_frame: start_frame + 10] = x_axis
    label[1, event_class, start_frame: start_frame + 10] = y_axis
    label[2, event_class, start_frame: start_frame + 10] = z_axis

    return label


class label4ADPIT(object):
    def __init__(self, num_axis, num_class, num_frame_wide):
        super().__init__()
        self._label_wide_0 = np.zeros([num_axis, num_class, num_frame_wide])  # a0----
        self._label_wide_1 = np.zeros([num_axis, num_class, num_frame_wide])  # --b0--
        self._label_wide_2 = np.zeros([num_axis, num_class, num_frame_wide])  # --b1--
        self._label_wide_3 = np.zeros([num_axis, num_class, num_frame_wide])  # ----c0
        self._label_wide_4 = np.zeros([num_axis, num_class, num_frame_wide])  # ----c1
        self._label_wide_5 = np.zeros([num_axis, num_class, num_frame_wide])  # ----c2

    def add_label_each_frame(self, list_time_array4frame_event, start_frame):
        if len(list_time_array4frame_event) == 1:
            self._label_wide_0 = add_label_each_frame(self._label_wide_0, list_time_array4frame_event[0], start_frame)
        elif len(list_time_array4frame_event) == 2:
            self._label_wide_1 = add_label_each_frame(self._label_wide_1, list_time_array4frame_event[0], start_frame)
            self._label_wide_2 = add_label_each_frame(self._label_wide_2, list_time_array4frame_event[1], start_frame)
        else:  # more than ov2
            self._label_wide_3 = add_label_each_frame(self._label_wide_3, list_time_array4frame_event[0], start_frame)
            self._label_wide_4 = add_label_each_frame(self._label_wide_4, list_time_array4frame_event[1], start_frame)
            self._label_wide_5 = add_label_each_frame(self._label_wide_5, list_time_array4frame_event[2], start_frame)

    def concat(self, index_diff, num_frame):
        label = np.stack((
            self._label_wide_0[:, :, index_diff: index_diff + num_frame],
            self._label_wide_1[:, :, index_diff: index_diff + num_frame],
            self._label_wide_2[:, :, index_diff: index_diff + num_frame],
            self._label_wide_3[:, :, index_diff: index_diff + num_frame],
            self._label_wide_4[:, :, index_diff: index_diff + num_frame],
            self._label_wide_5[:, :, index_diff: index_diff + num_frame]
        ))
        return label


def get_label(train_wav_length, time_array, start_sec, class_num):
    num_axis = 3  # X, Y, Z
    num_class = class_num
    num_frame = round(train_wav_length * 100) + 1
    label = np.zeros([num_axis, num_class, num_frame])

    end_sec = start_sec + train_wav_length

    index_diff = int(math.modf(start_sec * 10)[0] * 10)  # get second decimal place
    num_frame_wide = (int(np.ceil(end_sec * 10)) - int(np.floor(start_sec * 10)) + 1) * 10
    # "+ 1" is buffer for numerical error, such as index_diff=3 and num_frame_wide=130

    label_class = label4ADPIT(num_axis, num_class, int(num_frame_wide))

    for index, frame in enumerate(range(int(np.floor(start_sec * 10)), int(np.ceil(end_sec * 10)))):
        time_array4frame = time_array[time_array[:, 0] == frame]  # (0, 5) shape is ok
        sorted_time_array4frame = time_array4frame[np.argsort(time_array4frame[:, 1])]

        list_time_array4frame_event = []
        for i in range(len(sorted_time_array4frame)):
            list_time_array4frame_event.append(sorted_time_array4frame[i])
            if i == len(sorted_time_array4frame) - 1:  # if the last
                label_class.add_label_each_frame(list_time_array4frame_event, index * 10)
                list_time_array4frame_event = []
            elif sorted_time_array4frame[i, 1] != sorted_time_array4frame[i + 1, 1]:  # if the next is not the same class
                label_class.add_label_each_frame(list_time_array4frame_event, index * 10)
                list_time_array4frame_event = []

        label = label_class.concat(int(index_diff), num_frame)

    return label
