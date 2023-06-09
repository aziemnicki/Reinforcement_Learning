import numpy as np
from TD3 import TD3
from DDPG import DDPG
from utils import NaivePrioritizedBuffer
import os
import gym
import math
import argparse
import torch
from car_racing import Env

def test(env):
    ######### Hyperparameters #########
    env_name = env
    random_seed = 0
    filename = "DDPG_{}_{}".format(env_name, random_seed)
    directory = "./{}".format(env_name)  # save trained models
    img_stack = 4  # number of image stacks together
    action_repeat = 8  # repeat action in N frames
    max_episodes = 10

    ###################################

    env = Env(env_name, random_seed, img_stack, action_repeat)

    # training procedure:
    for episode in range(1, max_episodes + 1):
        state = env.reset()
        episode_timesteps = 0
        score = 0
        done = 0
        action_dim = env.action_space.shape[0]
        # select action and add exploration noise:
        while not done:
            policy = DDPG(action_dim, img_stack)
            policy.load(directory, filename)
            action = policy.select_action(state)
            action = action.clip(env.action_space.low, env.action_space.high)

            # take action in env:
            next_state, reward, done, die = env.step(action * np.array([2., 1., 1.]) + np.array([-1., 0., 0.]))
            env.render()
            state = next_state

            score += reward
            episode_timesteps += 1


if __name__ == "__main__":
    test('CarRacing-v2')
