# Copyright 2023 Sony Group Corporation.

class ValidationMonitor(object):
    def __init__(self, writer):
        self._writer = writer

    def add(self, i, val_results):
        all_test_metric = val_results[0]
        val_loss = val_results[1]

        self._writer.add_scalar('Metrics/1_ER-LD', all_test_metric[0], i)
        self._writer.add_scalar('Metrics/2_F-LD', all_test_metric[1], i)
        self._writer.add_scalar('Metrics/3_LE-CD', all_test_metric[2], i)
        self._writer.add_scalar('Metrics/4_LR-CD', all_test_metric[3], i)
        self._writer.add_scalar('Metrics/0_SELD-error', all_test_metric[4], i)

        self._writer.add_scalar('Loss/val', val_loss, i)
