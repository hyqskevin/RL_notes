# -*- coding: utf-8 -*-
# @Time    : 2021/8/10 1:05 PM
# @Author  : kevin_w
# @Site    : 
# @File    : util.py
# @Comment :

import torch
import argparse
import gym
import os
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--mode', default='train', type=str)  # mode = 'train' or 'test'

# OpenAI gym environment name, # ['BipedalWalker-v2', 'Pendulum-v0'] or any continuous environment
parser.add_argument("--env_name", default="Pendulum-v0")
# smoothing coefficient
parser.add_argument('--tau', default=0.005, type=float)
parser.add_argument('--target_update_interval', default=1, type=int)
parser.add_argument('--test_iteration', default=10, type=int)
parser.add_argument('--learning_rate', default=1e-3, type=float)
# discounted factor
parser.add_argument('--gamma', default=0.99, type=int)
# replay buffer size
parser.add_argument('--capacity', default=1000000, type=int)
# mini batch size
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--seed', default=True, type=bool)
parser.add_argument('--random_seed', default=65376, type=int)

# optional parameters
parser.add_argument('--sample_frequency', default=2000, type=int)
parser.add_argument('--render', default=False, type=bool)  # show UI or not
parser.add_argument('--log_interval', default=100, type=int)
parser.add_argument('--load', default=False, type=bool)  # load saved model
parser.add_argument('--render_interval', default=100, type=int)  # after render_interval, the env.render() will work
parser.add_argument('--exploration_noise', default=0.1, type=float)
parser.add_argument('--max_episode', default=10000, type=int)  # num of episode of games
parser.add_argument('--print_log', default=5, type=int)
parser.add_argument('--update_iteration', default=200, type=int)
args = parser.parse_args()


device = 'cuda' if torch.cuda.is_available() else 'cpu'
script_name = os.path.basename(__file__)
env = gym.make(args.env_name)

if args.seed:
    env.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)