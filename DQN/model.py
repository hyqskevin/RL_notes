# -*- coding: utf-8 -*-
# @Time    : 2021/7/24 7:19 PM
# @Author  : kevin_w
# @Site    : 
# @File    : model.py
# @Comment : define target net, policy net, loss function, optimizer
#            define sample action, store transition, optimize strategy

import layer
import memory
import math
from itertools import count
import random
import torch.nn as nn
import torch.optim as optim
from util import *


class DQN:

    def __init__(self):
        super(DQN, self).__init__()

        self.target_net = layer.Net(screen_height, screen_width, n_actions)
        self.policy_net = layer.Net(screen_height, screen_width, n_actions)
        # self.target_net.load_state_dict(self.policy_net.state_dict())
        # self.target_net.eval()
        self.optimizer = optim.RMSprop(self.policy_net.parameters())
        self.criterion = nn.SmoothL1Loss()
        # loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))
        self.memory_buffer = memory.ReplayMemory(MAX_MEMORY)
        self.steps_done = 0

    # epsilon-greedy or a = π(a|s)
    def select_action(self, state):
        sample = random.random()
        # eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        #                 math.exp(-1. * self.steps_done / EPS_DECAY)
        eps = EPSILON - self.steps_done * 0.01
        if eps < 0.1:
            eps = 0.1
        self.steps_done += 1

        if sample > eps:
            # use policy network to choose a*
            with torch.no_grad():
                policy_action = self.policy_net(state).max(1)[1].view(1, 1)
                # print('policy_action', policy_action)
                return policy_action
        else:
            greedy_action = torch.tensor([[random.randrange(n_actions)]], dtype=torch.long)
            # print('greedy_action', greedy_action)
            return greedy_action

    # store transition to memory
    # def store_transition(self, transition):
    #     self.memory_buffer.push(transition)

    def optimize_model(self):

        # collect enough experience data
        if len(self.memory_buffer) < BATCH_SIZE:
            return

        # get train batch
        transitions = self.memory_buffer.sample(BATCH_SIZE)
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                batch.next_state)), dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                           if s is not None])

        # get s,a,r
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Compute Q(s_t, a) - the model computes Q(s_t), then select the columns of actions taken.
        # torch.gather select action in terms of columns in action_batch
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(BATCH_SIZE)
        next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * GAMMA) + reward_batch

        # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            # clamp the gradient to prevent explosion
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()


def train():
    model = DQN()
    reward_list = []

    for i_episode in range(EPISODES):

        # Initialize the environment, state
        env.reset()
        last_screen = get_screen()
        current_screen = get_screen()
        state = current_screen - last_screen

        ep_reward = 0

        for t in range(1000):
            # Select and perform an action
            action = model.select_action(state)
            # _, reward, done, _ = env.step(action.item())
            # reward = torch.tensor([reward])

            next_state_data, _, done, _ = env.step(action.item())
            x, x_dot, theta, theta_dot = next_state_data
            reward = round(reward_func(env, x, x_dot, theta, theta_dot), 4)
            ep_reward += reward
            reward = torch.FloatTensor([reward])

            # Observe new state
            last_screen = current_screen
            current_screen = get_screen()

            if not done:
                next_state = current_screen - last_screen
            else:
                next_state = None

            # Store the transition in memory
            model.memory_buffer.push(state, action, next_state, reward)

            # Move to the next state
            state = next_state

            # Perform one step of the optimization (on the policy network)
            model.optimize_model()

            print("episodes {}, step is {}, reward {} ".format(i_episode, t, reward))

            if done:
                break

        reward_list.append(ep_reward)
        # Update the target network every n steps, copying all weights and biases in DQN
        if i_episode % TARGET_UPDATE == 0:
            model.target_net.load_state_dict(model.policy_net.state_dict())

    print('reward list', reward_list)
    print('Complete')
    env.render()
    env.close()
