import gym
from gym import spaces
from gym.utils import seeding

from .gym_input import *
from .env_variables import *
from .firelfy_task import Model

from numpy import pi
import numpy as np


class FireflyEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30
    }

    def __init__(self):
        #super(self.__class__, self).__init__()
        low = np.append([0., -pi, -1., -1., 0.], -10*np.ones(16))
        high = np.append([10., pi, 1., 1., 100.], 10*np.ones(16))
        #low[-1], high[-1] = 0., 100.
        self.action_space = spaces.Box(-np.ones(2), np.ones(2), dtype=np.float32)
        self.observation_space = spaces.Box(low, high, dtype=np.float32)

        self.model = Model(n1, n2, gains, obs_gains, log_rew_width)
        self.action_dim = self.model.action_dim
        self.state_dim = self.model.state_dim
        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        a = torch.Tensor([action[0], action[1]])
        state, reward, done, info = self.model(a)
        state = state.detach().numpy()
        reward = reward[0].item()
        done = done[0].item()
        return state, reward, done, info

    def reset(self, init=None):
        state = self.model.reset(init)
        return state.detach().numpy()

    def render(self, mode='human'):
        self.model.render()
