import torch
import numpy as np
from numpy import pi

from DDPGv2 import Agent, Noise
#from DDPGvtest import Agent, Noise
#from DDPGv2.utils import shrink
#from NAF import Agent, Noise

from FireflyEnv import Model
from FireflyEnv.gym_input import true_params

from shutil import copyfile
from collections import deque
rewards = deque(maxlen=100)

batch_size = 64
num_episodes = 2000

true_params = [p.data.clone() for p in true_params]
#env = Model(n1, n2, gains, obs_gains, log_rew_width)
env = Model(*true_params)
#env.Bstep.gains.data.copy_(torch.ones(2))
state_dim = env.state_dim
action_dim = env.action_dim
num_steps = int(env.episode_len)

std = 0.4
noise = Noise(action_dim, mean=0., std=std)
agent = Agent(state_dim, action_dim, hidden_dim=128, tau=0.001)
#agent.load('pretrained/ddpg/best_ddpg_model_7.pth.tar')
agent.load('pretrained/ddpg_new/ddpg_model_10.pth.tar')
#agent.load('pretrained/ddpg_circle/best_circle_model_2.pth.tar')
"""
best_avg = -90.
best_file = agent.file.split('/')
best_file[-1] = 'best_' + best_file[-1]
best_file = '/'.join(best_file)
print(best_file)
"""
env.Bstep.box = 1.
for episode in range(num_episodes):
    state = env.reset().view(1, -1)
    episode_reward = 0.
    std -= 1e-3
    if episode > 200:
        env.Bstep.box += 3e-3
    std = max(0.05, std)
    noise.reset(0., std)
    avg_rew = -100
    for t in range(num_steps):
        action = agent.select_action(state, noise)
        next_state, reward, done, info = env(action.view(-1))
        mask = 1 - done.float()
        next_state = next_state.view(1, -1)
        episode_reward += reward[0].item()
        #
        agent.memory.push(state, action, mask, next_state, reward)
        if len(agent.memory) > 500:
            #ploss, vloss = agent.learn(epochs=2, batch_size=batch_size)
            agent.learn(epochs=2, batch_size=batch_size)
        #
        state = next_state
        if done:
            break
        #
    #
    rewards.append(episode_reward)
    avg_rew = np.mean(rewards)
    print("Ep: {}, steps: {}, n: {:0.2f}, rew: {:0.4f}, avg_rew: {:0.4f}".format(episode, t+1, noise.scale, rewards[-1], np.mean(rewards)))
    if episode%25 == 0 and episode != 0:
        agent.save()
        """
        avg_rew = np.mean(rewards)
        if avg_rew > best_avg:
            best_avg = avg_rew
            copyfile(agent.file, best_file)
        """
