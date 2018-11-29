import torch
from numpy import pi
#n1 = -2.3 * torch.ones(5)
#n1 = -3.2 * torch.ones(5)
n1 = -4.6 * torch.ones(5)
n1[2] = -3.0 #-4.0
#n2 = torch.Tensor([-2.3, -6.3, -1.6, -1.6])
n2 = -2.3 * torch.ones(4)
gains = torch.Tensor([2., pi/2-0.3])
obs_gains = torch.ones(2)
log_rew_width = -2.8 * torch.ones(1) #-0.69
true_params = (n1, n2, gains, obs_gains, log_rew_width)
