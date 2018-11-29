import torch

def rewardFunc(rew_param, x, P, t, a, scale=100):
    #R = torch.diag(rew_param) # reward function is gaussian
    R = torch.eye(2) * rew_param
    P = P[:2, :2] # cov
    invP = torch.inverse(P)
    invS = torch.inverse(R) + invP
    S = torch.inverse(invS)
    mu = x[:2] # pos
    alpha = -0.5 * mu.matmul(invP - invP.mm(S).mm(invP)).matmul(mu)
    reward = torch.exp(alpha) * torch.sqrt(torch.det(S)/torch.det(P))
    reward = scale * reward # adjustment for reward per timestep
    return reward
