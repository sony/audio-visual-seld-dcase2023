# Copyright 2023 Sony Group Corporation.

import argparse
import os


def file_path(string):
    if os.path.isfile(string):
        return string
    else:
        raise FileNotFoundError(string)


def dir_path(string):
    if os.path.isdir(string):
        return string
    else:
        raise NotADirectoryError(string)


def model_monitor_path(string):
    if string == './data_dcase2023_task3/model_monitor':  # default is OK even if not a dir
        return string
    else:
        dir_path(string)


def get_args():
    parser = argparse.ArgumentParser()

    # setup
    parser.add_argument('--train', '-train', action='store_true', help='Train.')
    parser.add_argument('--val', '-val', action='store_true', help='Val.')
    parser.add_argument('--eval', '-eval', action='store_true', help='Eval.')
    parser.add_argument('--monitor-path', '-m', type=model_monitor_path, default='./data_dcase2023_task3/model_monitor', help='Path monitoring logs saved.')
    parser.add_argument('--random-seed', '-rs', type=int, default=0, help='Seed number for random and np.random.')
    # task
    parser.add_argument('--class-num', type=int, default=13, help='Total number of target classes, 13 is default for DCASE2023 Task3.')
    parser.add_argument('--train-wav-txt', '-twt', type=file_path, default='./data_dcase2023_task3/list_dataset/dcase2023t3_foa_devtrain_audiovisual.txt', help='Train wave file list text.')
    parser.add_argument('--val-wav-txt', '-valwt', type=file_path, default='./data_dcase2023_task3/list_dataset/dcase2023t3_foa_devtest.txt', help='Val wave file list text.')
    parser.add_argument('--eval-wav-txt', '-evalwt', type=file_path, default=None, help='Eval wave file list text.')
    parser.add_argument('--eval-model', '-em', type=file_path, default=None, help='Eval model.')
    # net
    parser.add_argument('--net', '-n', default='crnn', choices=['crnn'], help='Neural network architecture.')
    # optimizer
    parser.add_argument('--batch-size', '-b', type=int, default=16)
    parser.add_argument('--learning-rate', '-l', type=float, default=0.001)
    parser.add_argument('--weight-decay', '-w', type=float, default=0.000001, help='Weight decay factor of SGD update.')
    parser.add_argument('--max-iter', '-i', type=int, default=10000, help='Max iteration of training.')
    parser.add_argument('--model-save-interval', '-s', type=int, default=1000, help='The interval of saving model parameters.')
    # feature
    parser.add_argument('--sampling-frequency', '-fs', type=int, default=24000, help='Sampling frequency.')
    parser.add_argument('--feature', default='amp_phasediff', choices=['amp_phasediff'], help='Input audio feature type.')
    parser.add_argument('--fft-size', type=int, default=512, help='FFT size.')
    parser.add_argument('--stft-hop-size', type=int, default=240, help='STFT hop size.')
    parser.add_argument('--train-wav-length', type=float, default=1.27, help='Train wav length [seconds].')
    parser.add_argument('--eval-wav-hop-length', type=float, default=1.2, help='Eval wav hop length [seconds].')
    parser.add_argument('--feature-config', type=file_path, default='./feature/feature.json', help='config file is required for feature.')
    # threshold
    parser.add_argument('--threshold-config', type=file_path, default='./util/threshold.json', help='config file is required for threshold.')

    args = parser.parse_args()

    return args
