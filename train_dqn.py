import torch
import numpy as np
from numpy import pi

from DQN import Agent, Noise, mapping, inv_mapping
from FireflyEnv import Model
from FireflyEnv.gym_input import true_params

from collections import deque
rewards = deque(maxlen=100)

batch_size = 64
num_episodes = 2000

true_params = [item for item in true_params]
env = Model(*true_params)
state_dim = env.state_dim
action_dim = 7**2#action_map.size(0)
dlevels = 7
num_steps = int(env.episode_len)

std = 0.6
temp= 1.0
noise = Noise(2, mean=0., std=std)
agent = Agent(state_dim, action_dim, hidden_dim=64, tau=0.001, memory_size=1e6)
#agent.load('pretrained/dqn/25dqn_model_1.pth.tar')
#agent.load('pretrained/dqn/9dqn_model_4.pth.tar') temp=1-0.875

for episode in range(num_episodes):
    state = env.reset().view(1, -1)
    episode_reward = 0.
    std -= 1e-3
    #temp -=1e-3
    std = max(0.05, std)
    temp = max(temp, 0.05)
    noise.reset(0., std)
    #
    for t in range(num_steps):
        action = agent.select_action(state, temp, noise)
        action_vec = mapping(action, dlevels=dlevels)
        #
        #print(action_vec)
        next_state, reward, done, info = env(action_vec.view(-1))
        mask = 1 - done.float()
        next_state = next_state.view(1, -1)
        episode_reward += reward[0].item()
        #
        agent.memory.push(state, action, mask, next_state, reward)
        if len(agent.memory) > 500:
            agent.learn(epochs=2, batch_size=batch_size)
        #
        state = next_state
        if done:
            break
    rewards.append(episode_reward)
    #if episode % 50 == 49:
    #    agent.copy_weights()
    if episode % 20 == 19:
        agent.save()
    print("Ep: {}, steps: {}, n: {:0.2f}, rew: {:0.4f}, avg_rew: {:0.4f}".format(episode, t+1, std, rewards[-1], np.mean(rewards)))
