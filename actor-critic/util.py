# -*- coding: utf-8 -*-
# @Time    : 2021/8/4 5:12 PM
# @Author  : kevin_w
# @Site    : 
# @File    : util.py
# @Comment :

import numpy as np
import gym
import argparse
import torch
import torchvision.transforms as T
from collections import namedtuple
from PIL import Image
import matplotlib.pyplot as plt


# define parameter
parser = argparse.ArgumentParser(description="Actor Critic example")
parser.add_argument('--seed', type=int, default=1, metavar='seed')
parser.add_argument('--learning_rate', type=float, default=0.01, metavar='lr')
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
    # calculate Angle
    # theta has more proportion than threshold
    r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.5
    r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
    reward = r1 + r2
    return reward


def plot_training(steps):
    plt.figure(figsize=(8, 6))
    ax = plt.subplot(111)
    ax.cla()
    ax.grid()
    ax.set_title('Training')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Run Time')
    ax.plot(steps)
    RunTime = len(steps)

    path = './AC_CartPole-v0/' + str(RunTime) + '.jpg'
    if len(steps) % 100 == 0:
        plt.savefig(path)
    plt.pause(0.0000001)


# get state space, action space, \epsilon, saved action-value tuple
_, _, screen_height, screen_width = get_screen().shape
action_space = env.action_space.n
eps = np.finfo(np.float32).eps.item()
save_action = namedtuple('SavedAction', ['log_prob', 'value'])
