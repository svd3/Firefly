import torch
import numpy as np
from numpy import pi

from DDPGv2 import Agent, Noise
from DDPGv2.utils import shrink

from FireflyEnv import Model, pos_init
from FireflyEnv.gym_input import true_params

true_params = [p.data.clone() for p in true_params]

env = Model(*true_params)
state_dim = env.state_dim
action_dim = env.action_dim
num_steps = int(env.episode_len)

noise = Noise(action_dim, mean=0., std=0.05)
agent = Agent(state_dim, action_dim, hidden_dim=64, tau=0.001)
agentc = Agent(state_dim, action_dim, hidden_dim=64, tau=0.001)

agent.load('pretrained/ddpg/best_ddpg_model_7.pth.tar')
agentc.load('pretrained/ddpg_circle/best_circle_model_1.pth.tar')

for i in range(10):
    coord = list(pos_init(3.))
    coord[1] = -(pi/2) * torch.ones(1)
    r, ang, rel_ang = coord
    pos0 = r * torch.cat([torch.cos(ang), torch.sin(ang)])
    x = -rel_ang
    R = torch.stack([torch.cat([torch.cos(x), -torch.sin(x)]), torch.cat([torch.sin(x), torch.cos(x)])])
    #
    state = env.reset(coord)
    traj1 = []
    for t in range(num_steps):
        traj1.append(env.x)
        action = agent.select_action(state, noise)
        next_state, reward, done, info = env(action.view(-1))
        mask = 1 - done.float()
        state = next_state
        if info['stop']:
            break
    #
    traj1 = torch.stack(traj1).detach()
    #
    state = env.reset(coord)
    traj2 = []
    for t in range(num_steps):
        traj2.append(env.x)
        action = agentc.select_action(state, noise)
        next_state, reward, done, info = env(shrink(action).view(-1))
        mask = 1 - done.float()
        state = next_state
        if info['stop']:
            break
    #
    traj2 = torch.stack(traj2).detach()
    #
    pts1 = (R.matmul((traj1[:,:2] - pos0).t())).numpy()
    ptsc = (R.matmul((traj2[:,:2] - pos0).t())).numpy()
    #
    plt.plot(pts1[0], pts1[1], color=cmap(0), linestyle = '--')
    plt.plot(ptsc[0], ptsc[1], color=cmap(1), linestyle = '-.')

plt.axis([-2,2, 0,3])
plt.axis('equal')
plt.show()
