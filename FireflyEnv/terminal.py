import torch

def is_terminal(x, a, goal_radius, terminal_vel):
    """
    x is just the mean of the belief state
    """
    ## stopped and reached_target are correlated
    scale = (goal_radius/terminal_vel)**2
    mu = torch.cat((x[:2], x[-2:]))
    Q = torch.diag(torch.Tensor([1, 1, scale, scale]))

    r = torch.sqrt(mu.matmul(Q).matmul(mu))
    if r <= goal_radius:
        return torch.ByteTensor([True]), True

    return torch.ByteTensor([False]), False

def is_terminal_velocity(x, a, goal_radius, terminal_vel):
    """
    x is just the mean of the belief state
    """
    pos = x[:2]
    vels = x[-2:]
    reached_target = (torch.norm(pos) <= goal_radius)

    if torch.norm(vels) <= terminal_vel:
        return torch.ByteTensor([True]), reached_target

    return torch.ByteTensor([False]), False

def is_terminal_action(x, a, goal_radius, terminal_vel):
    """
    x is just the mean of the belief state
    """
    pos = x[:2]
    reached_target = (torch.norm(pos) <= goal_radius)

    if torch.norm(a) <= terminal_vel:
        return torch.ByteTensor([True]), reached_target

    return torch.ByteTensor([False]), False
