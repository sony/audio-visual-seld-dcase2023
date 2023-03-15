# Copyright 2023 Sony Group Corporation.

import torch
import torch.nn as nn
import torch.nn.functional as F
import json

from net.net_util import interpolate


def create_net_seld(args):
    with open(args.feature_config, 'r') as f:
        feature_config = json.load(f)
    if args.net == 'crnn':
        Net = AudioVisualCRNN(class_num=args.class_num,
                              in_channels=feature_config[args.feature]["ch"])
    return Net


class AudioVisualCRNN(nn.Module):
    def __init__(self, class_num, in_channels, interp_ratio=16):
        super().__init__()
        self.class_num = class_num
        self.interp_ratio = interp_ratio

        # Audio
        aud_embed_size = 64
        self.audio_encoder = CNN3(in_channels=in_channels, out_channels=aud_embed_size)

        # Visual
        vis_embed_size = 64
        vis_in_size = 2 * 6 * 37
        project_vis_embed_fc1 = nn.Linear(vis_in_size, vis_embed_size)
        project_vis_embed_fc2 = nn.Linear(vis_embed_size, vis_embed_size)
        self.vision_encoder = nn.Sequential(project_vis_embed_fc1,
                                            project_vis_embed_fc2)

        # Audio-Visual
        in_size_gru = aud_embed_size + vis_embed_size
        self.gru = nn.GRU(input_size=in_size_gru, hidden_size=256,
                          num_layers=1, batch_first=True, bidirectional=True)
        self.fc_xyz = nn.Linear(512, 3 * 3 * self.class_num, bias=True)

    def forward(self, x_a, x_v):
        x_a = x_a.transpose(2, 3)
        b_a, c_a, t_a, f_a = x_a.size()  # input: batch_size, mic_channels, time_steps, freq_bins
        b, c, t, f = b_a, c_a, t_a, f_a
        x_a = self.audio_encoder(x_a)
        x_a = torch.mean(x_a, dim=3)  # x_a: batch_size, feature_maps, time_steps

        x_v = x_v.view(x_v.size(0), -1)
        x_v = self.vision_encoder(x_v)
        x_v = torch.unsqueeze(x_v, dim=-1).repeat(1, 1, 8)  # repeat for time_steps

        x = torch.cat((x_a, x_v), 1)

        x = x.transpose(1, 2)  # x: batch_size, time_steps, feature_maps
        self.gru.flatten_parameters()
        (x, _) = self.gru(x)

        x = self.fc_xyz(x)  # event_output: batch_size, time_steps, 3 * 3 * class_num
        x = interpolate(x, self.interp_ratio)
        x = x.transpose(1, 2)
        x = x.view(-1, 3, 3, self.class_num, t)

        return x


class CNN3(nn.Module):
    def __init__(self, in_channels, out_channels,
                 kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=kernel_size, stride=stride,
                               padding=padding, bias=False)
        self.conv2 = nn.Conv2d(in_channels=out_channels,
                               out_channels=out_channels,
                               kernel_size=kernel_size, stride=stride,
                               padding=padding, bias=False)
        self.conv3 = nn.Conv2d(in_channels=out_channels,
                               out_channels=out_channels,
                               kernel_size=kernel_size, stride=stride,
                               padding=padding, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.bn3 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = F.relu_(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, kernel_size=(4, 4))
        x = F.relu_(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, kernel_size=(2, 4))
        x = F.relu_(self.bn3(self.conv3(x)))
        x = F.max_pool2d(x, kernel_size=(2, 2))
        return x
