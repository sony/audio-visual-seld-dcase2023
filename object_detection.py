# Copyright 2023 Sony Group Corporation.

import torch
import numpy as np
from mmdet.apis import init_detector, inference_detector


class ObjectDetection(object):
    def __init__(self):
        self._n_box = 6  # magic number for max number of boxes
        self._len_dist = 36  # magic number as a distribution is plotted from 0 to _len_dist
        self._thresh_conf = 0.10  # magic number for threshold of box confidence

        config_file = "./mmdetection/yolox_tiny_8x8_300e_coco.py"
        checkpoint_file = "./mmdetection/yolox_tiny_8x8_300e_coco_20211124_171234-b4047906.pth"
        self._model = init_detector(config_file, checkpoint_file, device="cuda:0" if torch.cuda.is_available() else "cpu")

    def get_dist_shape(self):
        dist_shape = (2, 6, 37)
        return dist_shape

    def img2box(self, img_in):
        img = img_in[:, :, [2, 1, 0]]  # BGR for MMDetection
        result = inference_detector(self._model, img)
        box_out = np.zeros((self._n_box, 4))
        i = 0
        for object_xyxy in result[0]:  # 0: person
            if object_xyxy[4] > self._thresh_conf:
                box_out[i, 0] = object_xyxy[0] / 360  # 0 - 1
                box_out[i, 1] = object_xyxy[1] / 180
                box_out[i, 2] = object_xyxy[2] / 360
                box_out[i, 3] = object_xyxy[3] / 180
                i = i + 1
                if i >= self._n_box:
                    break
        return box_out

    def _func_Gauss(self, x, mu, sigma):
        return np.exp(-(x - mu)**2 / sigma**2)

    def box2dist(self, box_in):
        points = np.arange(0, self._len_dist + 1, 1)
        dist_azi = np.zeros((self._n_box, self._len_dist + 1))
        dist_ele = np.zeros((self._n_box, self._len_dist + 1))
        for i, each_box in enumerate(box_in):
            if np.sum(each_box) > 0:
                center_azi = (each_box[0] + each_box[2]) / 2  # 0 - 1
                center_ele = (each_box[1] + each_box[3]) / 2
                len_azi = each_box[2] - each_box[0]  # 0 - 1
                len_ele = each_box[3] - each_box[1]
                dist_azi[i] = self._func_Gauss(points, center_azi * self._len_dist, len_azi / 2 * self._len_dist)
                dist_ele[i] = self._func_Gauss(points, center_ele * self._len_dist, len_ele / 2 * self._len_dist)
        dist_out = np.stack((dist_azi, dist_ele), axis=0)
        return dist_out
