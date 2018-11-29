import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

def t(A):
    return A.transpose(-1, -2)

class NAFQNet(nn.Module):
    def __init__(self, input_dim, action_dim, hidden_dim=128):
        super(self.__class__, self).__init__()
        self.input_dim = input_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim

        num_outputs = action_dim

        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.mu = nn.Linear(hidden_dim, num_outputs)
        self.V = nn.Linear(hidden_dim, 1)
        self.L = nn.Linear(hidden_dim, num_outputs ** 2)
        self.cholesky = torch.distributions.transforms.LowerCholeskyTransform()

    def forward(self, states, actions):
        x = torch.relu(self.linear1(states))
        x = torch.relu(self.linear2(x))
        mu = torch.tanh(self.mu(x))
        V = self.V(x)
        Q = None
        if actions is not None:
            L = torch.tanh(self.L(x)).view(-1, self.action_dim, self.action_dim)
            L = self.cholesky(L)
            P = L.bmm(t(L))
            a_mu = (actions - mu).unsqueeze(2)
            A = -0.5 * 10 * t(a_mu).bmm(P).bmm(a_mu).squeeze(2)
            Q = A + V
        return mu, Q, V
