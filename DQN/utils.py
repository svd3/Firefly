import os, sys

import torch
from torch.autograd import Variable
from numpy import pi

def mapping(index, dlevels):
    index = int(index)
    r, c = index//dlevels, index%dlevels
    v = -1 + r*2/(dlevels - 1)
    w = -1 + c*2/(dlevels - 1)
    return torch.Tensor([v, w])

def inv_mapping(action, dlevels):
    invres = (dlevels - 1)/2
    v, w = torch.clamp(torch.round(invres * action)/invres, -1, 1)
    r = torch.round((v + 1) * invres)
    c = torch.round((w + 1) * invres)
    return int(r), int(c), torch.tensor(int(dlevels*r + c))

class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = None

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout = self._original_stdout

def variable(x, **kwargs):
    if torch.cuda.is_available():
        return Variable(x, **kwargs).cuda()
    return Variable(x, **kwargs)

def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)

def next_path(path_pattern):
    """
    path_pattern = 'file-%s.txt':
    """
    i = 1
    while os.path.exists(path_pattern % i):
        i = i * 2

    a, b = (i // 2, i)
    while a + 1 < b:
        c = (a + b) // 2 # interval midpoint
        a, b = (c, b) if os.path.exists(path_pattern % c) else (a, c)

    return path_pattern % b
