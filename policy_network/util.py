# -*- coding: utf-8 -*-
# @Time    : 2021/8/2 5:12 PM
# @Author  : kevin_w
# @Site    : 
# @File    : util.py
# @Comment :

import numpy as np
import gym
import argparse
import torch
import torchvision.transforms as T
from PIL import Image

# define parameter
parser = argparse.ArgumentParser(description="RL REINFORCE example")
parser.add_argument('--seed', type=int, default=1, metavar='seed')
parser.add_argument('--gamma', type=float, default=0.99, metavar='gamma')
parser.add_argument('--batch_size', type=int, default=128, metavar='batch')
parser.add_argument('--episode', type=int, default=1000, metavar='episode')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='interval between training status logs (default: 10)')

args = parser.parse_args()

resize = T.Compose([
    T.ToPILImage(),
    T.Resize(40, interpolation=Image.CUBIC),
    T.ToTensor()
])

env = gym.make('CartPole-v0').unwrapped
env.reset()
env.seed(args.seed)
torch.manual_seed(args.seed)


def get_cart_location(screen_width):
    # x_threshold: max coordinate of angle
    width = env.x_threshold * 2
    scale = screen_width / width
    return int(env.state[0] * scale + screen_width / 2.0)


def get_screen():
    # screen 800x1200x3 -> 3x800x1200 Color x Height x Width
    screen = env.render(mode='rgb_array').transpose((2, 0, 1))
    _, screen_height, screen_width = screen.shape
    # clip height [0.4, 0.6]
    screen = screen[:, int(screen_height * 0.4): int(screen_height * 0.8)]
    # clip width
    view_width = int(screen_width * 0.6)
    cart_location = get_cart_location(screen_width)
    if cart_location < view_width // 2:
        slice_range = slice(view_width)
    elif cart_location > (screen_width - view_width // 2):
        slice_range = slice(-view_width, None)
    else:
        slice_range = slice(cart_location - view_width // 2,
                            cart_location + view_width // 2)
    screen = screen[:, :, slice_range]
    screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
    screen = resize(torch.from_numpy(screen)).unsqueeze(0)

    return screen


def reward_func(x, theta):
    # calculate Angle at which will fail the episode
    r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.5
    r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
    reward = r1 + r2
    return reward

