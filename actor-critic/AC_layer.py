# -*- coding: utf-8 -*-
# @Time    : 2021/8/4 5:04 PM
# @Author  : kevin_w
# @Site    : 
# @File    : AC_layer.py
# @Comment :

import torch.nn as nn


def param_size(size, kernel_size=5, stride=2):
    return (size - (kernel_size - 1) - 1) // stride + 1


# define REINFORCE nn
class ActorCritic(nn.Module):
    def __init__(self, height, width, output_size):
        super(ActorCritic, self).__init__()

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
        self.linear_1 = nn.Linear(input_size, 256)
        self.linear_2 = nn.Linear(256, output_size)
        self.action_output = nn.Softmax(dim=-1)
        self.state_value_output = nn.Linear(256, 1)

        self.rewards = []
        self.actions = []

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)  # flat to 1D
        x = self.linear_1(x)

        # get one state value
        state_value= self.state_value_output(x)
        # get softmax action output
        action_fc = self.linear_2(x)
        action = self.action_output(action_fc)
        return action, state_value

