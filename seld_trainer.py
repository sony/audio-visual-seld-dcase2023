# Copyright 2023 Sony Group Corporation.

import torch
import torch.nn
import torch.optim as optim

from seld_data_loader import create_data_loader
from net.net_seld import create_net_seld


class SELDTrainer(object):
    def __init__(self, args, object_detection):
        self._args = args

        self._device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self._data_loader = create_data_loader(self._args, object_detection)

        self._net = create_net_seld(self._args)
        self._net.to(self._device)
        self._net.train()

        self._criterion = MSELoss_ADPIT()
        self._optimizer = optim.Adam(
            self._net.parameters(),
            lr=self._args.learning_rate,
            weight_decay=self._args.weight_decay
        )

    def receive_input(self):
        _input_a, _input_v, _label, _ = next(iter((self._data_loader)))

        self._input_a = _input_a.to(self._device)
        self._input_v = _input_v.to(self._device)
        self._label = _label.to(self._device)

    def back_propagation(self):
        self._net.train()
        self._optimizer.zero_grad()

        self._output = self._net(self._input_a, self._input_v)
        self._loss = self._criterion(self._output, self._label)
        self._loss.backward()

        self._optimizer.step()

    def save(self, each_monitor_path=None, iteration=None, start_time=None):
        self._each_checkpoint_path = '{}/params_{}_{:07}.pth'.format(
            each_monitor_path,
            start_time,
            iteration)
        torch_net_state_dict = self._net.state_dict()
        checkpoint = {'model_state_dict': torch_net_state_dict,
                      'optimizer_state_dict': self._optimizer.state_dict(),
                      'rng_state': torch.get_rng_state(),
                      'cuda_rng_state': torch.cuda.get_rng_state()}
        torch.save(checkpoint, self._each_checkpoint_path)
        print('save checkpoint to {}.'.format(self._each_checkpoint_path))

    def get_loss(self):
        return self._loss.cpu().detach().numpy()

    def get_lr(self):
        return self._optimizer.state_dict()['param_groups'][0]['lr']

    def get_each_model_path(self):
        return self._each_checkpoint_path


class MSELoss_ADPIT(object):
    def __init__(self):
        super().__init__()
        self._each_loss = torch.nn.MSELoss(reduction='none')

    def _each_calc(self, output, target):
        return self._each_loss(output, target).mean(dim=(1, 2))  # class-wise frame-level

    def __call__(self, output, target):
        """
        Auxiliary Duplicating Permutation Invariant Training (ADPIT) for 13 (=1+6+6) possible combinations
        Args:
            output: [batch_size, num_track=3, num_axis=3, num_class=13, num_frames]
            target:  [batch_size, num_track=6, num_axis=3, num_class=13, num_frames]
        Return:
            loss: scalar
        """
        target_A0 = target[:, 0, :, :, :]  # A0, no ov from the same class, [batch_size, num_axis=3, num_class=13, num_frames]
        target_B0 = target[:, 1, :, :, :]  # B0, ov with 2 sources from the same class
        target_B1 = target[:, 2, :, :, :]  # B1
        target_C0 = target[:, 3, :, :, :]  # C0, ov with 3 sources from the same class
        target_C1 = target[:, 4, :, :, :]  # C1
        target_C2 = target[:, 5, :, :, :]  # C2

        target_A0A0A0 = torch.stack((target_A0, target_A0, target_A0), 1)  # 1 permutation of A (no ov from the same class), [batch_size, num_track=3, num_axis=3, num_class=13, num_frames]
        target_B0B0B1 = torch.stack((target_B0, target_B0, target_B1), 1)  # 6 permutations of B (ov with 2 sources from the same class)
        target_B0B1B0 = torch.stack((target_B0, target_B1, target_B0), 1)
        target_B0B1B1 = torch.stack((target_B0, target_B1, target_B1), 1)
        target_B1B0B0 = torch.stack((target_B1, target_B0, target_B0), 1)
        target_B1B0B1 = torch.stack((target_B1, target_B0, target_B1), 1)
        target_B1B1B0 = torch.stack((target_B1, target_B1, target_B0), 1)
        target_C0C1C2 = torch.stack((target_C0, target_C1, target_C2), 1)  # 6 permutations of C (ov with 3 sources from the same class)
        target_C0C2C1 = torch.stack((target_C0, target_C2, target_C1), 1)
        target_C1C0C2 = torch.stack((target_C1, target_C0, target_C2), 1)
        target_C1C2C0 = torch.stack((target_C1, target_C2, target_C0), 1)
        target_C2C0C1 = torch.stack((target_C2, target_C0, target_C1), 1)
        target_C2C1C0 = torch.stack((target_C2, target_C1, target_C0), 1)

        pad4A = target_B0B0B1 + target_C0C1C2
        pad4B = target_A0A0A0 + target_C0C1C2
        pad4C = target_A0A0A0 + target_B0B0B1
        loss_0 = self._each_calc(output, target_A0A0A0 + pad4A)  # padded with target_B0B0B1 and target_C0C1C2 in order to avoid to set zero as target
        loss_1 = self._each_calc(output, target_B0B0B1 + pad4B)  # padded with target_A0A0A0 and target_C0C1C2
        loss_2 = self._each_calc(output, target_B0B1B0 + pad4B)
        loss_3 = self._each_calc(output, target_B0B1B1 + pad4B)
        loss_4 = self._each_calc(output, target_B1B0B0 + pad4B)
        loss_5 = self._each_calc(output, target_B1B0B1 + pad4B)
        loss_6 = self._each_calc(output, target_B1B1B0 + pad4B)
        loss_7 = self._each_calc(output, target_C0C1C2 + pad4C)  # padded with target_A0A0A0 and target_B0B0B1
        loss_8 = self._each_calc(output, target_C0C2C1 + pad4C)
        loss_9 = self._each_calc(output, target_C1C0C2 + pad4C)
        loss_10 = self._each_calc(output, target_C1C2C0 + pad4C)
        loss_11 = self._each_calc(output, target_C2C0C1 + pad4C)
        loss_12 = self._each_calc(output, target_C2C1C0 + pad4C)

        loss_min = torch.min(
            torch.stack((loss_0,
                         loss_1,
                         loss_2,
                         loss_3,
                         loss_4,
                         loss_5,
                         loss_6,
                         loss_7,
                         loss_8,
                         loss_9,
                         loss_10,
                         loss_11,
                         loss_12), dim=0),
            dim=0).indices

        loss = (loss_0 * (loss_min == 0) +
                loss_1 * (loss_min == 1) +
                loss_2 * (loss_min == 2) +
                loss_3 * (loss_min == 3) +
                loss_4 * (loss_min == 4) +
                loss_5 * (loss_min == 5) +
                loss_6 * (loss_min == 6) +
                loss_7 * (loss_min == 7) +
                loss_8 * (loss_min == 8) +
                loss_9 * (loss_min == 9) +
                loss_10 * (loss_min == 10) +
                loss_11 * (loss_min == 11) +
                loss_12 * (loss_min == 12)).mean()

        return loss
