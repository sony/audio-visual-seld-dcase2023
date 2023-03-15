# Copyright 2023 Sony Group Corporation.

import os
import datetime
import json
import numpy as np
import random
from torch.utils.tensorboard import SummaryWriter
import tqdm
from collections import OrderedDict
import time

from util.args import get_args
from util.validation_monitor import ValidationMonitor
from object_detection import ObjectDetection
from seld_trainer import SELDTrainer
from seld_validator import SELDValidator


def main():
    args = get_args()
    if args.eval:
        evaluation(args)
    elif args.train:
        train(args)


def train(args):
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)

    start_time = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
    each_monitor_path = '{}/{}'.format(args.monitor_path, start_time)
    os.makedirs(each_monitor_path, exist_ok=True)

    object_detection = ObjectDetection()
    seld_trainer = SELDTrainer(args, object_detection)
    seld_validator = SELDValidator(args, object_detection, each_monitor_path)

    writer = SummaryWriter(log_dir=each_monitor_path)
    monitor_val = ValidationMonitor(writer)
    with open(os.path.join(each_monitor_path, 'args.json'), 'x') as fout:
        json.dump(vars(args), fout, indent=4)

    with tqdm.tqdm(range(1, args.max_iter + 1), desc='[Train]') as pbar:
        for i in pbar:
            seld_trainer.receive_input()
            seld_trainer.back_propagation()
            writer.add_scalar('Loss/train', seld_trainer.get_loss(), i)
            writer.add_scalar('Optimizer/lr', seld_trainer.get_lr(), i)
            pbar.set_postfix(OrderedDict(loss=seld_trainer.get_loss()))

            if i % args.model_save_interval == 0:
                seld_trainer.save(each_monitor_path, i, start_time)
                if args.val:
                    val_results = seld_validator.validation(seld_trainer.get_each_model_path())
                    monitor_val.add(i, val_results)

    time.sleep(0.01)  # wait for TensorBoard writing of the last iteration


def evaluation(args):
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)

    object_detection = ObjectDetection()
    seld_validator = SELDValidator(args, object_detection, os.path.dirname(args.eval_model))
    _ = seld_validator.validation(args.eval_model)


if __name__ == '__main__':
    main()
