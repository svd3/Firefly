from .agent import DQNAgent as Agent
from .utils import mapping, inv_mapping
from .noise import Noise, OUNoise

import torch
import numpy as np

x = np.round(np.arange(-1, 1.01, 1), 1)
x, y = np.meshgrid(x, x)
action_map = np.vstack([x.ravel(), y.ravel()]).T
action_map = torch.Tensor(action_map) * 0.1
