import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

class QNet(nn.Module):
    def __init__(self, input_dim, n_actions, hidden_dim=128):
        super(self.__class__, self).__init__()
        self.input_dim = input_dim
        self.action_dim = n_actions
        self.hidden_dim = hidden_dim

        num_outputs = n_actions

        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, hidden_dim)
        self.linear4 = nn.Linear(hidden_dim, n_actions)
        self.qvalues = nn.Linear(n_actions, num_outputs)

    def forward(self, inputs):
        x = torch.relu(self.linear1(inputs))
        x = torch.relu(self.linear2(x))
        x = torch.relu(self.linear3(x))
        x = torch.relu(self.linear4(x))
        qvalues = self.qvalues(x)
        return qvalues

class Policy(nn.Module):
    def __init__(self, n_actions):
        super(self.__class__, self).__init__()

        self.action_dim = n_actions
        self.temperature = nn.Linear(1, 1, bias=False) # scaling
        self.temperature.weight.data = torch.ones(1, 1)

    def forward(self, x):
        return torch.softmax(self.temperature(x), dim=len(x.size())-1)
