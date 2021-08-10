# REINFORCE on the CartPole-v0 task

# -*- coding: utf-8 -*-
# @Time    : 2021/8/2 4:28 PM
# @Author  : kevin_w
# @Site    :
# @File    : util.py
# @Comment :


from itertools import count

import numpy as np
import torch.optim as optim
from torch.distributions import Categorical
from nn_layer import REINFORCE
from util import *


# select action policy log(pi(\theta))
def select_action(state, policy):
    prob = policy(state)
    c = Categorical(prob)
    sample_action = c.sample()
    policy.probs_log.append(c.log_prob(sample_action))
    return sample_action.item()


# finish episode
def loss_function(policy):
    R = 0
    policy_loss = []
    rewards = []
    eps = np.finfo(np.float32).eps.item()

    # get discount reward in each episode, from t-1 to 0
    for r in policy.rewards[::-1]:
        # R_{t+1} = r + \gamma R_{t}
        R = r + args.gamma * R
        rewards.insert(0, R)
    rewards = torch.tensor(rewards)
    # normalization
    rewards = (rewards - rewards.mean()) / (rewards.std() + eps)

    # policy loss: - \sum log(prob)*reward
    for prob, reward in zip(policy.probs_log, rewards):
        policy_loss.append(-prob * reward)
    policy_loss = torch.cat(policy_loss).sum()

    return policy_loss


# train
def train():

    # init env, policy nn optimizer
    _, _, screen_height, screen_width = get_screen().shape
    n_actions = env.action_space.n
    policy = REINFORCE(screen_height, screen_width, n_actions)
    optimizer = optim.Adam(policy.parameters(), lr=0.01)

    for episode in count(1):
        env.reset()
        last_screen = get_screen()
        current_screen = get_screen()
        state = current_screen - last_screen
        # collect trajectory
        for i in range(10000):
            env.render()
            # random sample action
            action = select_action(state, policy)
            # get reward and next state
            next_state, _, done, _ = env.step(action)
            x, _, theta, _ = next_state
            reward = round(reward_func(x, theta), 4)
            policy.rewards.append(reward)

            # Observe new state
            last_screen = current_screen
            current_screen = get_screen()

            if not done:
                next_state = current_screen - last_screen
            else:
                next_state = None
            state = next_state

            if done:
                break

        # update policy: \alpha * policy_loss
        optimizer.zero_grad()
        policy_loss = loss_function(policy)
        policy_loss.backward()
        optimizer.step()

        eps_rewards = np.sum(policy.rewards)
        # print info
        if episode % args.log_interval == 0:
            print('episode {}: last time {}, reward {}, loss{}'.format(
                episode, i, eps_rewards, policy_loss
            ))

        # clean rewards and probs_log every episode
        policy.rewards = []
        policy.probs_log = []

    env.close()

    # openAI gym test
    # env = gym.make('CartPole-v0').unwrapped
    # for episode in range(10):
    #     env.reset()
    #     for _ in range(1000):
    #         env.render()
    #         next_state_data, _, done, _ = env.step(env.action_space.sample())
    #         if done:
    #             break
    # env.close()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    train()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
