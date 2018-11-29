import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

from .rewards import *
from .terminal import *
from .belief_step import BeliefStep

from .env_utils import *
from .env_variables import *
from .plotter import Render

class Model(nn.Module):
    def __init__(self, n1, n2, gains, obs_gains, log_reward_width):
        super(self.__class__, self).__init__()
        # constants
        self.dt = DELTA_T
        self.action_dim = ACTION_DIM
        self.goal_radius = GOAL_RADIUS
        self.terminal_vel = TERMINAL_VEL
        self.episode_len = EPISODE_LEN
        self.episode_time = EPISODE_LEN * self.dt

        self.Bstep = BeliefStep(n1, n2, gains, obs_gains, self.dt)
        self.reward_param = Parameter(log_reward_width)

        self.rendering = Render()

        self.reset()

    def reset(self, init=None):
        self.time = torch.zeros(1)
        if init is None:
            init = pos_init(self.Bstep.box)
        r, ang, rel_ang = init
        pos = torch.cat([r*torch.cos(ang), r*torch.sin(ang)])
        ang = ang + pi + rel_ang
        ang = range_angle(ang)
        vels = torch.zeros(2)
        self.x = torch.cat([pos, ang, vels])
        relx = torch.cat([r, rel_ang, vels])

        # covariance
        self.P = torch.eye(5) * 1e-8
        vecL = vectorLowerCholesky(self.P)

        state = torch.cat([relx, self.time, vecL])
        self.state_dim = state.size(0)
        return state

    def forward(self, a, Y=None):
        x, P, time = self.x, self.P, self.time
        time += self.dt
        x, P, _ = self.Bstep(x, P, a, Y)

        pos, ang, vels = x[:2], x[2], x[-2:]
        r = torch.norm(pos).view(-1)
        rel_ang = ang - torch.atan2(-pos[1], -pos[0]).view(-1)
        rel_ang = range_angle(rel_ang)
        relx = torch.cat([r, rel_ang, vels])
        vecL = vectorLowerCholesky(P)
        state = torch.cat([relx, time, vecL])

        terminal = self._isTerminal(x, a)
        done = time >= self.episode_time
        #reward = terminal * self._get_reward(x, P, time, a) - 1
        reward = -1 * torch.ones(1) #- 0.1 * vels.norm()**2
        self.x, self.P, self.time = x, P, time
        if terminal:
            reward = reward + self._get_reward(x, P, time, a)
            state = self.reset()
        return state, reward.view(-1), done, {'stop': terminal}

    def _get_state(self):
        x, P, time = self.x, self.P, self.time
        pos, ang, vels = x[:2], x[2], x[-2:]
        r = torch.norm(pos).view(-1)
        rel_ang = ang - torch.atan2(-pos[1], -pos[0]).view(-1)
        rel_ang = range_angle(rel_ang)
        relx = torch.cat([r, rel_ang, vels])
        vecL = vectorLowerCholesky(P)
        state = torch.cat([relx, time, vecL])
        return state

    def _get_reward(self, x, P, time, a):
        rew_param = torch.exp(self.reward_param)
        reward = rewardFunc(rew_param, x, P, time, a, scale=10)
        return reward

    def _isTerminal(self, x, a, log=True):
        goal_radius = self.goal_radius
        terminal_vel = self.terminal_vel
        #terminal, reached_target = is_terminal_velocity(x, a, goal_radius, terminal_vel)
        terminal, reached_target = is_terminal_action(x, a, goal_radius, terminal_vel)
        if terminal and log:
            #print("Stopped. {:0.2f}".format(torch.norm(x[:2]).item()))
            pass
        if reached_target and log:
            print("Goal!!")
        return terminal.item() == 1

    def _single_step(self, state, time, a):
        relx = state[:4]
        r, rel_ang, vels = relx[0], relx[1], relx[-2:]
        ang = torch.zeros(1).uniform_(-pi, pi)
        pos = torch.cat([r*torch.cos(ang), r*torch.sin(ang)])
        ang = ang + pi + rel_ang
        ang = range_angle(ang)
        x = torch.cat([pos, ang, vels])

        vecL = state[4:]
        P = inverseCholesky(vecL)
        x, P, _ = self.Bstep(x, P, a)
        time += self.dt

        pos, ang, vels = x[:2], x[2], x[-2:]
        r = torch.norm(pos).view(-1)
        rel_ang = ang - torch.atan2(-pos[1], -pos[0]).view(-1)
        relx = torch.cat([r, rel_ang, vels])
        vecL = vectorLowerCholesky(P)
        state = torch.cat([relx, vecL])

        terminal = self._isTerminal(x, a)
        done = time >= self.episode_time
        reward = terminal * self._get_reward(x, P, time, a)  -1.

        return state, reward, time, done

    def render(self):
        goal = torch.zeros(2)
        self.rendering.render(goal, self.x, self.P)
