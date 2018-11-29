import gym
import FireflyEnv
from FireflyEnv.env_variables import EPISODE_LEN

import time
import torch
import numpy as np
from numpy import pi
from DDPGv2 import Agent, Noise
#from NAF import Agent, Noise
#from DQN import Agent, Noise, mapping

from collections import deque
rewards = deque(maxlen=100)

num_episodes = 10
env = gym.make('FireflyTorch-v0')
state_dim = env.state_dim
action_dim = env.action_dim
num_steps = int(EPISODE_LEN)

std = 0.05
noise = Noise(action_dim, mean=0., std=std)
agent = Agent(state_dim, action_dim, hidden_dim=64, tau=0.001)
#agent.load('pretrained/naf/naf_model_2.pth.tar')
#agent.load('pretrained/ddpg/best_ddpg_model_7.pth.tar')
#agent.load('pretrained/best_stop_model_2.pth.tar')
agent.load('pretrained/ddpg_new/ddpg_model_11.pth.tar')
#agent.load('pretrained/dqn/dqn_model_2.pth.tar')

for episode in range(num_episodes):
    state = torch.Tensor([env.reset()])
    episode_reward = 0.
    for t in range(num_steps):
        action = agent.select_action(state, noise)
        #action = torch.clamp(action, -1, 1)
        #action = mapping(action, 5)
        #action = agent.select_action(state, 0.1)
        #action = action_map[action]
        #print(action)
        next_state, reward, done, info = env.step(action.view(-1))
        time.sleep(0.005)
        if info['stop']:
            time.sleep(0.5)
        env.render()
        mask = 1 - done
        next_state = torch.Tensor([next_state])
        episode_reward += reward
        state = next_state
        if done:
            break
    rewards.append(episode_reward)
    print("Ep: {}, steps: {}, n: {:0.2f}, rew: {:0.4f}, avg_rew: {:0.4f}".format(episode, t+1, std, rewards[-1], np.mean(rewards)))

print("Done")
