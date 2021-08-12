# -*- coding: utf-8 -*-
# @Time    : 2021/8/4 4:25 PM
# @Author  : kevin_w
# @Site    : 
# @File    : actor-critic.py
# @Comment :


import os
import numpy as np
import torch.optim as optim
from itertools import count
import torch.nn as nn
from torch.distributions import Categorical
from AC_layer import ActorCritic
from util import *

# torch.set_default_tensor_type(torch.FloatTensor)


def select_action(state, model):
    prob, state_value = model(state)

    prob = prob.to(torch.float64)[0]
    state_value = state_value.to(torch.float64)[0]

    c = Categorical(prob)
    sample_action = c.sample()
    model.actions.append(save_action(c.log_prob(sample_action), state_value))
    return sample_action.item()


def loss_function(model):
    R = 0
    saved_actions = model.actions
    policy_loss = []
    value_loss = []
    rewards = []
    criterion = nn.SmoothL1Loss()

    for r in model.rewards[::-1]:
        R = r + args.gamma * R
        rewards.insert(0, R)

    rewards = torch.tensor(rewards)
    rewards = (rewards - rewards.mean()) / (rewards.std() + eps)

    for(log_prob, state_value), U in zip(saved_actions, rewards):
        # value loss
        value_loss.append(criterion(state_value, torch.tensor([U])))
        # policy loss Q(S, A), loss = - Q(S, A) * log(\pi(A|S))
        # reward = r - log_prob
        # policy loss U - V(s), loss = - (U - V(S)) * log(\pi(A|S))
        reward = U - state_value.item()
        policy_loss.append((-log_prob * reward))

    return policy_loss, value_loss


def train():
    model = ActorCritic(screen_height, screen_width, action_space)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    live_time = []

    for episode in count():
        env.reset()
        last_screen = get_screen()
        current_screen = get_screen()
        state = current_screen - last_screen

        for t in count():
            env.render()
            action = select_action(state, model)
            next_state, _, done, _ = env.step(action)
            x, _, theta, _ = next_state
            reward = round(reward_func(x, theta), 4)

            model.rewards.append(reward)

            # Observe new state
            last_screen = current_screen
            current_screen = get_screen()

            if not done:
                next_state = current_screen - last_screen
            else:
                next_state = None

            state = next_state

            if done or t >= 1000:
                break

        # plot last time when training
        live_time.append(t)
        plot_training(live_time)

        # if episode % 200 == 0:
        #     modelPath = 'Times.pkl'
        #     torch.save(model, modelPath)

        policy_loss, value_loss = loss_function(model)
        optimizer.zero_grad()
        loss = (torch.stack(policy_loss).sum() + torch.stack(value_loss).sum()) / t
        loss.backward()
        optimizer.step()

        eps_rewards = np.sum(model.rewards)
        print('episode {}: last time {}, reward {}, loss{}'.format(
            episode, t, eps_rewards, loss
        ))

        # clean rewards and probs_log every episode
        model.rewards = []
        model.actions = []


if __name__ == '__main__':
    os.makedirs('./AC_CartPole-v0', exist_ok=True)
    train()



