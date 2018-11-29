import numpy as np

class Noise(object):
    def __init__(self, action_dim, mean=0., std=1.):
        self.mu = np.ones(action_dim) * mean
        self.scale = std
        self.action_dim = action_dim

    def reset(self, mean, std):
        self.mu = np.ones(self.action_dim) * mean
        self.scale = std

    def noise(self):
        n = np.random.randn(2)
        return self.mu + self.scale*n

class OUNoise(object):
    def __init__(self, action_dim, scale=0.1, mu=0, theta=0.15, sigma=0.2):
        self.action_dimension = action_dim
        self.scale = scale
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.action_dimension) * self.mu
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_dimension) * self.mu

    def noise(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return self.state * self.scale
