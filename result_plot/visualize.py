import time
import torch
import numpy as np
from numpy import pi
from DDPGv2 import Agent, Noise

from FireflyEnv import Model
from FireflyEnv.gym_input import true_params
from FireflyEnv.env_utils import inverseCholesky, ellipse, pos_init

import matplotlib.pyplot as plt

env = Model(*true_params)
env2 = Model(*true_params)
#env.Bstep.gains.data.copy_(torch.ones(2))
state_dim = env.state_dim
action_dim = env.action_dim
num_steps = int(env.episode_len)

agent = Agent(state_dim, action_dim, hidden_dim=64, tau=0.001)
#agent.load('pretrained/ddpg/best_ddpg_model_6.pth.tar')
agent.load('pretrained/stop_model_2.pth.tar')
noise = Noise(action_dim, mean=0., std=0.01)

def R(x):
    R = np.eye(2)
    R[0,0], R[0,1] = np.cos(x), -np.sin(x)
    R[1,0], R[1,1] = np.sin(x), np.cos(x)
    return R

def qvalue(state, action):
    return agent.critic(state.view(1,-1), action.view(1,-1))

def getQmatrix(state, ndis=100):
    x = np.linspace(-1, 1, ndis)
    W, V = np.meshgrid(x, x)
    W, V = torch.FloatTensor(W), torch.FloatTensor(V)
    A = torch.stack([V.view(-1), W.view(-1)]).t()
    S = state.view(1,-1).repeat(A.shape[0], 1)
    Q = agent.critic(S, A).view(ndis, ndis).detach()
    return Q, V, W

def opt_value(state):
    Q, _, _ = getQmatrix(state, 100)
    return Q.max()

def getDecision(state, ndis=10):
    #vels = state[2:4].detach()
    x = np.linspace(0, 3, ndis)
    X, Y = np.meshgrid(x, x)
    X, Y = torch.FloatTensor(X), torch.FloatTensor(Y)
    D = torch.zeros(ndis**2)
    i=0
    for x, y in zip(X.view(-1), Y.view(-1)):
        coord =  (torch.sqrt(x**2 + y**2).view(-1), torch.zeros(1), -torch.atan2(y, x).view(-1) + pi/4)
        _state = env2.reset(coord) # change this, write custom reset function or initialize another env
        #_state[2:4] = vels
        D[i] = opt_value(_state)
        i += 1
    D = D.view(ndis, ndis)
    return D, X, Y

colors = plt.cm.get_cmap('Set2')
coord = pos_init(3)
coord[0][0] = 3
state = env.reset(coord)
posvec = env.x
env.P = 2*env.P
for i in range(5):
    action = agent.select_action(state, noise)
    next_state, reward, done, info = env(action.view(-1)*torch.tensor([1., 1]))
    #env.P = 1.1*env.P
    env.x[:2] = 1.02*env.x[:2]
    next_state = env._get_state()
    if info['stop']:
        print("break")
        break
    state = next_state

Q,V, W = getQmatrix(state)
plt.contourf(W, V, Q, levels = torch.linspace(Q.min(), Q.max(), 100), cmap='summer')

def do():
    state = env._get_state()
    D, X, Y = getDecision(state, 10)
    Df = D - opt_value(state)
    vmin, vmax = -150, 150
    plt.contourf(X, Y, Df, levels=torch.linspace(vmin, vmax,21), vmin=vmin, vmax=vmax, cmap='summer')
    plt.colorbar()
    c = plt.contour(X, Y, Df, levels=torch.linspace(vmin, vmax,21), vmin=vmin, vmax=vmax, cmap='gray_r')
    plt.clabel(c, inline=1, fontsize=9, fmt='%1.1f')
    plt.axes().set_aspect('equal')
    posvec = env.x
    r, ang = state[0].data.numpy(), pi/4 - state[1].data.numpy()
    x, y = r*np.cos(ang), r*np.sin(ang)
    vecL = state[5:]
    xang = pi/4 - posvec[2].data.numpy()
    P = inverseCholesky(vecL).data.numpy()[:2, :2]
    _R = R(xang)
    P = _R.dot(P).dot(_R.T)
    pts = np.vstack(ellipse(np.array([x,y]), P, conf_int=5.991**2))
    plt.scatter(x, y, c=colors(5))
    plt.plot(pts[0], pts[1], c=colors(5), ls='--')
    plt.show(0)
