import gym
import numpy as np
from ppo_torch import Agent

if __name__ == '__main__':
    env = gym.make('CartPole-v1',render_mode="human")
    N = 20
    batch_size = 5
    n_epochs = 4
    alpha = 0.0003
    agent = Agent(n_actions=env.action_space.n, batch_size=batch_size,
                    alpha=alpha, n_epochs=n_epochs,
                    input_dims=env.observation_space.shape)

    agent.load_models()
    episodes = 5
    for episode in range(1, episodes + 1):
        observation, info = env.reset()
        x = False
        score = 0

        while not x:
            action, prob, val = agent.choose_action(observation)
            observation_, reward, x, y, info = env.step(action)
            score += reward
            observation = observation_

        print(f'Episode:{episode} Score:{score}')
    env.close()
