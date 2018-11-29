import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from .env_utils import *
from .env_variables import *

class BeliefStep(nn.Module):
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

        #self.H_ = torch.zeros(2, 5)
        #self.H_[:, -2:] = torch.diag(self.obs_gains)
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
        A_ = torch.zeros(5, 5)
        A_[:3, :3] = torch.eye(3)
        A_[0, 2] = - vel * torch.sin(ang) * dt
        A_[1, 2] = vel * torch.cos(ang) * dt
        return A_

    def H(self, x):
        H_ = torch.zeros(2, 5)
        H_[:, -2:] = torch.diag(self.obs_gains)
        return H_

    def L(self, x, a, w, x_):
        dt = self.dt
        # one extra call to dynamics is necessary since noise isn't additive
        if not (w == 0).all():
            x_ = self.dynamics(x, a, w)
        px, py, ang, vel, ang_vel = x_
        L = torch.zeros(5, 5)
        L[0, 2] = -vel * torch.sin(ang) * dt
        L[0, 3] = torch.cos(ang) * dt
        L[0, 4] = -vel * torch.sin(ang) * dt * dt
        L[1, 2] = vel * torch.cos(ang) * dt
        L[1, 3] = torch.sin(ang) * dt
        L[1, 4] = vel * torch.cos(ang) * dt * dt
        L[2, 4] = dt
        return torch.eye(5) #+ L

    def M(self, x, v):
        # doesn't depend on noise (v)
        Kvels2 = (x[-2:]**2) * torch.exp(self.n2_kappa*2)
        sigma2 = torch.exp(self.n2_sigma*2)
        return torch.diag(torch.sqrt(Kvels2 + sigma2))

    def forward(self, x, P, a, Y=None):
        Q = torch.diag(torch.exp(self.n1*2))
        #R = torch.diag(torch.exp(self.n2*2))
        R = torch.eye(2)
        I = torch.eye(5)
        w = torch.exp(self.n1) * torch.randn(5)
        v_k = torch.exp(self.n2_kappa) * torch.randn(2)
        v_s = torch.exp(self.n2_sigma) * torch.randn(2)
        zeros2 = torch.zeros(2)

        x_ = self.dynamics(x, a, w*0)
        A = self.A(x_, a)   # A = self.A(x, a, w*0)
        H = self.H(x_)      # H = self.H(x_, v*0)
        L = self.L(x, a, w, x_)
        M = self.M(x_, zeros2)

        P_ = A.mm(P).mm(A.t()) + L.mm(Q).mm(L.t())
        if not is_pos_def(P_):
            print("P_:", P_)
            print("P:", P)
            print("A:", A)
            APA = A.mm(P).mm(A.t())
            print("APA:", APA)
            print("APA +:", is_pos_def(APA))

        S = H.mm(P_).mm(H.t()) + M.mm(R).mm(M.t())
        K =  P_.mm(H.t()).mm(torch.inverse(S))
        if Y is None:
            Y = self.observations(x_, v_s, v_k) # with noise

        x = x_ + K.matmul(Y - self.observations(x_, zeros2, zeros2))

        I_KH = I - K.mm(H)
        P = I_KH.mm(P_)
        P = (P + P.t())/2  + 1e-6 * I# make symmetric to avoid computational overflows
        if not is_pos_def(P):
            print("here")
            print("P:", P)
        return x, P, K
