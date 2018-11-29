import torch
import torch.nn as nn
import numpy as np
from numpy import pi


p = nn.Parameter(torch.rand(1))
s = nn.Parameter(torch.rand(1))
Q = nn.Sequential(nn.Linear(1, 16), nn.ReLU(), nn.Linear(16, 8), nn.ReLU(), nn.Linear(8, 1))

opt = torch.optim.Adam([p, s] + list(Q.parameters()), lr=1e-3)
for i in range(100):
    x = torch.randn(64, 1)
    err = (Q(x) - p * x/torch.exp(2*s) - 0.99 * Q(p * x))**2
    err = err.mean()
    err.backward()
    #print(err.item())
    opt.step()
    opt.zero_grad()
