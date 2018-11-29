import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

from .nets import NAFQNet
from .utils import *
from .ReplayMemory import ReplayMemory

CUDA = torch.cuda.is_available()

class Agent():
    def __init__(self, input_dim, action_dim, hidden_dim=128, gamma=0.99, tau=0.01, memory_size=1e6):
        self.input_dim = input_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.gamma = gamma
        self.tau = tau
        print("Running NAF Agent")

        if CUDA:
            self.qnet = NAFQNet(input_dim, action_dim, hidden_dim).cuda()
            self._qnet = NAFQNet(input_dim, action_dim, hidden_dim).cuda()
        else:
            self.qnet = NAFQNet(input_dim, action_dim, hidden_dim)
            self._qnet = NAFQNet(input_dim, action_dim, hidden_dim)

        self.optim = Adam(self.qnet.parameters(), lr=1e-3)

        self.priority = False
        self.memory = ReplayMemory(int(memory_size), priority=self.priority)

        self.args = (input_dim, action_dim, hidden_dim)
        hard_update(self._qnet, self.qnet)
        self.create_save_file()

    def select_action(self,  state, exploration=None):
        mu, _, V = self.qnet(state, None)
        mu = mu.detach()
        if exploration is not None:
            if CUDA:
                mu += torch.Tensor(exploration.noise()).cuda()
            else:
                mu += torch.Tensor(exploration.noise())
        return mu.clamp(-1, 1)

    def update_parameters(self, batch):
        states = variable(torch.cat(batch.state))
        next_states = variable(torch.cat(batch.next_state))
        actions = variable(torch.cat(batch.action))
        rewards = variable(torch.cat(batch.reward).unsqueeze(1))
        masks = variable(torch.cat(batch.mask)).unsqueeze(1)
        with torch.no_grad():
            _, _, next_values =  self._qnet(next_states, None)
            target_qvalues = rewards + self.gamma * masks * next_values

        self.optim.zero_grad()
        _, pred_qvalues, _ = self.qnet(states, actions)
        loss = F.mse_loss(pred_qvalues, target_qvalues)
        loss.backward()
        nn.utils.clip_grad_norm_(self.qnet.parameters(), 1)
        self.optim.step()

    def learn(self, epochs, batch_size=64):
        for epoch in range(epochs):
            # sample new batch here
            batch, _ = self.memory.sample(batch_size)
            self.update_parameters(batch)
            soft_update(self._qnet, self.qnet, self.tau)

    def save(self):
        state = {
            'args': self.args,
            'qnet': self.qnet.state_dict(),
        }
        #'feature_dict': self.features.state_dict(),
        torch.save(state, self.file)
        print("Saved to " + self.file)

    def load(self, file='pretrained/naf/naf_model.pth.tar'):
        state = torch.load(file, map_location=lambda storage, loc: storage)
        if self.args != state['args']:
            print('Agent parameters from file are different from call')
            print('Overwriting agent to load file ... ')
            args = state['args']
            self.__init__(*args)

        self.qnet.load_state_dict(state['qnet'])
        hard_update(self._qnet, self.qnet)
        print('Loaded')
        return

    def create_save_file(self):
        path = './pretrained/naf'
        os.makedirs(path, exist_ok=True)
        self.file = next_path(path + '/' + 'naf_model_%s.pth.tar')
