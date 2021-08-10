# -*- coding: utf-8 -*-
# @Time    : 2021/8/2 9:04 AM
# @Author  : kevin_w
# @Site    : 
# @File    : nn_layer.py
# @Comment :

import torch.nn as nn
import torch.nn.functional as func


def param_size(size, kernel_size=5, stride=2):
    return (size - (kernel_size - 1) - 1) // stride + 1


# define REINFORCE nn
class REINFORCE(nn.Module):
    def __init__(self, height, width, output_size):
        super(REINFORCE, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, stride=2),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=5, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        conv_h = param_size(param_size(height))
        conv_w = param_size(param_size(width))
        input_size = conv_h * conv_w * 32
        self.linear = nn.Linear(input_size, output_size)
        self.output = nn.Softmax()

        self.rewards = []
        self.probs_log = []

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)  # flat to 1D
        x = self.linear(x)
        x = self.output(x)
        return x

