"""
@author: Viet Nguyen <nhviet1009@gmail.com>
"""
import torch.nn as nn


def param_size(size, kernel_size=5, stride=2):
    return (size - (kernel_size - 1) - 1) // stride + 1


class DeepQNetwork(nn.Module):
    def __init__(self, height, width, output_size):
        super(DeepQNetwork, self).__init__()
        # raise NotImplementedError
        # input 4x84x84
        self.conv = nn.Sequential(
            nn.Conv2d(4, 16, kernel_size=5, stride=2),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=5, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        # conv_h = param_size(param_size(height))
        # conv_w = param_size(param_size(width))
        # input_size = conv_h * conv_w * 64
        self.linear = nn.Linear(10368, output_size)

    def _create_weights(self):
        raise NotImplementedError

    def forward(self, x):
        # raise NotImplementedError
        x = self.conv(x)
        x = x.view(x.size(0), -1)  # flat to 1D
        x = self.linear(x)
        return x
