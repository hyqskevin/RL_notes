# -*- coding: utf-8 -*-
# @Time    : 2021/6/24 5:47 PM
# @Author  : kevin_w
# @Site    :
# @File    : memory.py
# @Comment : replay memory

# define the capacity, push/sample option and length
import random
from collections import namedtuple, deque


class ReplayMemory(object):
    def __init__(self, capacity):
        # define the max capacity of memory
        self.memory = deque(maxlen=capacity)
        self.Transition = namedtuple('Transition',
                                     ('state', 'action', 'reward', 'next_state'))

    def __len__(self):
        return len(self.memory)

    def push(self, *args):
        self.memory.append(self.Transition(*args))

    def sample(self, bach_size):
        return random.sample(self.memory, bach_size)
