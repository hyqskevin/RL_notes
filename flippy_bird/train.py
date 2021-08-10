"""
@author: Viet Nguyen <nhviet1009@gmail.com>
"""
import argparse
import os
import shutil
from random import random, randint, sample, randrange

import numpy as np
import torch
import torch.nn as nn
from tensorboardX import SummaryWriter

from src.deep_q_network import DeepQNetwork
from src.flappy_bird import FlappyBird
from src.utils import pre_processing

from collections import namedtuple, deque


class ReplayMemory(object):
    def __init__(self, capacity):
        # define the max capacity of memory
        self.memory = deque(maxlen=capacity)
        self.Transition = namedtuple('Transition',
                                     ('state', 'action', 'reward', 'next_state', 'terminal'))

    def __len__(self):
        return len(self.memory)

    def push(self, *args):
        self.memory.append(self.Transition(*args))

    def sample(self, bach_size):
        return sample(self.memory, bach_size)


def get_args():
    parser = argparse.ArgumentParser(
        """Implementation of Deep Q Network to play Flappy Bird""")
    parser.add_argument("--image_size", type=int, default=84, help="The common width and height for all images")
    parser.add_argument("--batch_size", type=int, default=64, help="The number of images per batch")
    parser.add_argument("--optimizer", type=str, choices=["sgd", "adam"], default="adam")
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--initial_epsilon", type=float, default=0.6)
    parser.add_argument("--final_epsilon", type=float, default=1e-4)
    parser.add_argument("--num_iters", type=int, default=200000)
    parser.add_argument("--replay_memory_size", type=int, default=5000,
                        help="Number of epoches between testing phases")
    parser.add_argument("--log_path", type=str, default="tensorboard")
    parser.add_argument("--saved_path", type=str, default="trained_models")

    args = parser.parse_args()
    return args


def train(opt):
    if torch.cuda.is_available():
        torch.cuda.manual_seed(123)
    else:
        torch.manual_seed(123)

    if os.path.isdir(opt.log_path):
        shutil.rmtree(opt.log_path)
    os.makedirs(opt.log_path)
    writer = SummaryWriter(opt.log_path)

    # load game image
    game_state = FlappyBird()
    image, reward, terminal = game_state.next_frame(0)
    image = pre_processing(image[:game_state.screen_width, :int(game_state.base_y)], opt.image_size, opt.image_size)
    image = torch.from_numpy(image)
    # print(image.shape)

    # define nn model & loss func
    height = width = image.shape[0]
    model = DeepQNetwork(height, width, output_size=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-6)
    criterion = nn.MSELoss()

    if torch.cuda.is_available():
        model.cuda()
        image = image.cuda()

    # 每次选取4张图片
    state = torch.cat(tuple(image for _ in range(4)))[None, :, :, :]
    replay_memory = ReplayMemory(capacity=opt.replay_memory_size)
    iter = 0
    epsilon = opt.initial_epsilon

    while iter < opt.num_iters:

        # 你可以在此：
        # 1。输出你模型的预测
        # 2。选择你的EPSILON， 输出变量名: epsilon
        # 3。使用 EPSILON greedy ，输出
        # TODO

        rand = random()
        epsilon = max(0.01, epsilon - (iter // 100) * 0.01)
        prediction = model(state)
        if rand > epsilon:
            with torch.no_grad():
                action = prediction.max(1)[1].view(1, 1)
        else:
            action = torch.tensor([[randrange(2)]])

        # get next_state
        next_image, reward, terminal = game_state.next_frame(action)
        next_image = pre_processing(next_image[:game_state.screen_width, :int(game_state.base_y)], opt.image_size,
                                    opt.image_size)
        next_image = torch.from_numpy(next_image)
        if torch.cuda.is_available():
            next_image = next_image.cuda()
        next_state = torch.cat((state[0, 1:, :, :], next_image))[None, :, :, :]

        # 把 state, action, reward, next_state, terminal 放在一个LIST 中，并加入到replay_memory 中
        # 如果 replay_memory 满了（> opt.replay_memory_size），请做相应处理（删除）
        # 从replay buffer 中sample 出BATCH 进行训练
        # TODO

        replay_memory.push(state, action, reward, next_state, terminal)
        batch = replay_memory.sample(min(len(replay_memory), opt.batch_size))

        # 对 sample 出来的结果进行拆分，获得s，a，r，s'
        state_batch, action_batch, reward_batch, next_state_batch, terminal_batch = replay_memory.Transition(*zip(*batch))
        state_batch = torch.cat(tuple(state for state in state_batch))
        action_batch = torch.from_numpy(
            np.array([[1, 0] if action == 0 else [0, 1] for action in action_batch], dtype=np.float32))
        reward_batch = torch.from_numpy(np.array(reward_batch, dtype=np.float32)[:, None])
        next_state_batch = torch.cat(tuple(state for state in next_state_batch))

        if torch.cuda.is_available():
            state_batch = state_batch.cuda()
            action_batch = action_batch.cuda()
            reward_batch = reward_batch.cuda()
            next_state_batch = next_state_batch.cuda()

        # get Q(s, a), Q(s', a)
        current_prediction_batch = model(state_batch)
        next_prediction_batch = model(next_state_batch)

        # get y = r + \gamma * max_a Q(s', a)
        y_batch = torch.cat(
            tuple(reward if terminal else reward + opt.gamma * torch.max(prediction) for reward, terminal, prediction in
                  zip(reward_batch, terminal_batch, next_prediction_batch)))

        # get q = Q(s,a)* chosen a
        # 选择已经选好的action值进行加和
        q_value = torch.sum(current_prediction_batch * action_batch, dim=1)
        optimizer.zero_grad()
        # y_batch = y_batch.detach()
        loss = criterion(q_value, y_batch)
        loss.backward()
        optimizer.step()

        state = next_state
        iter += 1
        print("Iteration: {}/{}, Action: {}, Loss: {}, Epsilon {}, Reward: {}, Q-value: {}".format(
            iter + 1,
            opt.num_iters,
            action,
            loss,
            epsilon, reward, torch.max(prediction)))
        writer.add_scalar('Train/Loss', loss, iter)
        writer.add_scalar('Train/Epsilon', epsilon, iter)
        writer.add_scalar('Train/Reward', reward, iter)
        writer.add_scalar('Train/Q-value', torch.max(prediction), iter)
        if (iter + 1) % 1000000 == 0:
            torch.save(model, "{}/flappy_bird_{}".format(opt.saved_path, iter + 1))
    torch.save(model, "{}/flappy_bird".format(opt.saved_path))


if __name__ == "__main__":
    opt = get_args()
    train(opt)
