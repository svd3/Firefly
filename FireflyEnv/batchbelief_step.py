import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from .env_utils import *
from .env_variables import *

def t(A):
    return A.transpose(-1, -2)

def _batch_inverse(bmat):
    r"""
    Returns the inverses of a batch of square matrices.
    """
    n = bmat.size(-1)
    flat_bmat_inv = torch.stack([m.inverse() for m in bmat.reshape(-1, n, n)])
    return flat_bmat_inv.reshape(bmat.shape)

class BatchBeliefStep(nn.Module):
    def __init__(self, n1, n2, gains, obs_gains, dt):
        super(self.__class__, self).__init__()

        # declare parameters
        self.n1 = Parameter(n1) # noise terms in dynamics
        self.n2_sigma = Parameter(n2[:2]) # noise terms in observations
        self.n2_kappa = Parameter(n2[2:]) # noise terms in observations
        self.gains = Parameter(gains)
        self.obs_gains = Parameter(obs_gains)

        self.dt = dt
        self.box = WORLD_SIZE
        return

    def dynamics(self, x, a, w):
        dt = self.dt
        g = self.gains
        px, py, ang, vel, ang_vel = x
        vel = 0.0 * vel + g[0] * a[0] + w[3]
        ang_vel = 0.0 * ang_vel + g[1] * a[1] + w[4]
        ang = ang + ang_vel * dt + w[2]
        ang = range_angle(ang)
        px = px + vel * torch.cos(ang) * dt + w[0]
        py = py + vel * torch.sin(ang) * dt + w[1]
        px = torch.clamp(px, -self.box, self.box)
        py = torch.clamp(py, -self.box, self.box)
        return torch.stack((px, py, ang, vel, ang_vel))

    def observations(self, x, v_s, v_k):
        og = self.obs_gains
        vel, ang_vel = x[-2:]
        vel = og[0] * vel + vel * v_k[0] + v_s[0]
        ang_vel = og[1] * ang_vel + ang_vel * v_k[1] + v_s[1]
        return torch.stack((vel, ang_vel))

    def A(self, x_, a):
        dt = self.dt
        px, py, ang, vel, ang_vel = x_
        N = x_.shape[1]
        A_ = torch.zeros(N, 5, 5)
        A_[:, :3, :3] = torch.eye(3).repeat(N, 1, 1)
        A_[:, 0, 2] = - vel * torch.sin(ang) * dt
        A_[:, 1, 2] = vel * torch.cos(ang) * dt
        return A_

    def H(self, x):
        N = x.shape[1]
        H_ = torch.zeros(N, 2, 5)
        H_[:, :, -2:] = torch.diag(self.obs_gains).repeat(N, 1, 1)
        return H_

    def L(self, x, a, w, x_):
        dt = self.dt
        # one extra call to dynamics is necessary since noise isn't additive
        if not (w == 0).all():
            x_ = self.dynamics(x, a, w)
        px, py, ang, vel, ang_vel = x_
        N = x.shape[1]
        L = torch.zeros(N, 5, 5)
        L[:, 0, 2] = -vel * torch.sin(ang) * dt
        L[:, 0, 3] = torch.cos(ang) * dt
        L[:, 0, 4] = -vel * torch.sin(ang) * dt * dt
        L[:, 1, 2] = vel * torch.cos(ang) * dt
        L[:, 1, 3] = torch.sin(ang) * dt
        L[:, 1, 4] = vel * torch.cos(ang) * dt * dt
        L[:, 2, 4] = dt
        return torch.eye(5).repeat(N, 1, 1) #+ L

    def M(self, x, v):
        # doesn't depend on noise (v)
        N = x.shape[1]
        Kvels2 = (x[-2:]**2) * torch.exp(self.n2_kappa*2).view(-1, 1)
        sigma2 = torch.exp(self.n2_sigma*2)
        D = torch.Tensor([[[1, 0], [0, 0]], [[0, 0], [0, 1]]])
        K = Kvels2.t().matmul(D).permute(1, 0, 2)
        M_ = K + torch.diag(sigma2).repeat(N, 1, 1)
        #return torch.diag(torch.sqrt(Kvels2 + sigma2))
        return torch.sqrt(M_)

    def forward(self, x, P, a, Y=None):
        x, a = x.t(), a.t()
        N = x.shape[1]
        Q = torch.diag(torch.exp(self.n1*2))
        #R = torch.diag(torch.exp(self.n2*2))
        R = torch.eye(2)
        I = torch.eye(5)
        w = torch.exp(self.n1).view(-1, 1) * torch.randn(5, N)
        v_k = torch.exp(self.n2_kappa).view(-1, 1) * torch.randn(2, N)
        v_s = torch.exp(self.n2_sigma).view(-1, 1) * torch.randn(2, N)
        zeros2 = torch.zeros(2, N)

        x_ = self.dynamics(x, a, w*0)
        A = self.A(x_, a)   # A = self.A(x, a, w*0)
        H = self.H(x_)      # H = self.H(x_, v*0)
        L = self.L(x, a, w, x_)
        M = self.M(x_, zeros2)

        P_ = A.bmm(P).bmm(t(A)) + L.matmul(Q).bmm(t(L))
        """
        if not is_pos_def(P_):
            print("P_:", P_)
            print("P:", P)
            print("A:", A)
            APA = A.mm(P).mm(A.t())
            print("APA:", APA)
            print("APA +:", is_pos_def(APA))
        """
        S = H.bmm(P_).bmm(t(H)) + M.matmul(R).bmm(t(M))
        K =  P_.bmm(t(H)).bmm(torch.inverse(S))
        if Y is None:
            Y = self.observations(x_, v_s, v_k) # with noise

        err = (Y - self.observations(x_, zeros2, zeros2)).t().unsqueeze(2)
        x = x_ + K.bmm(err).squeeze(2).t()

        I_KH = I - K.bmm(H)
        P = I_KH.bmm(P_)
        P = (P + t(P))/2  + 1e-6 * I # make symmetric to avoid computational overflows
        """
        if not is_pos_def(P):
            print("here")
            print("P:", P)
        """
        return x.t(), P, K
