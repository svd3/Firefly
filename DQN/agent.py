import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import numpy as np

from .nets import QNet, Policy
from .utils import *
from .ReplayMemory import ReplayMemory

CUDA = torch.cuda.is_available()

huber = nn.SmoothL1Loss()

class DQNAgent():
    def __init__(self, input_dim, n_actions, hidden_dim=128, gamma=0.99, tau=0.01, memory_size=1e6):
        self.input_dim = input_dim
        self.action_dim = n_actions
        self.hidden_dim = hidden_dim
        self.gamma = gamma
        self.tau = tau
        self.dlevels = int(np.sqrt(n_actions))

        if CUDA:
            self.qnet = QNet(input_dim, n_actions, hidden_dim).cuda()
            self._qnet = QNet(input_dim, n_actions, hidden_dim).cuda()
        else:
            self.qnet = QNet(input_dim, n_actions, hidden_dim)
            self._qnet = QNet(input_dim, n_actions, hidden_dim)

        self.optimizer = Adam(self.qnet.parameters(), lr=1e-3)

        self.priority = False
        self.memory = ReplayMemory(int(memory_size), priority=self.priority)

        self.args = (input_dim, n_actions, hidden_dim)
        #hard_update(self._qnet, self.qnet)  # Make sure target is with the same weight
        self.copy_weights()
        self.create_save_file()

    def select_action(self,  state, temp=1., exploration=None):
        qvalues = self.qnet(state)/temp
        dist = torch.distributions.Categorical(torch.softmax(qvalues, dim=1))
        action_index = dist.sample()
        if exploration is not None:
            mu = mapping(action_index, dlevels=self.dlevels)
            mu += torch.Tensor(exploration.noise())
            action_index = inv_mapping(mu, dlevels=self.dlevels)[2].view(-1)

        return action_index


    def update_parameters(self, batch):
        states = variable(torch.cat(batch.state))
        next_states = variable(torch.cat(batch.next_state))
        actions = variable(torch.cat(batch.action)).unsqueeze(1)
        rewards = variable(torch.cat(batch.reward).unsqueeze(1))
        masks = variable(torch.cat(batch.mask)).float()
        with torch.no_grad():
            #max_qvalues = masks * self._qnet(next_states).max(dim=1)[0]
            max_qvalues = self._qnet(next_states).max(dim=1)[0]
            #max_qvalues = torch.log(torch.sum(torch.exp(self._qnet(next_states))))
            target_qvalues = rewards + self.gamma * max_qvalues.view(-1, 1)

        self.optimizer.zero_grad()
        #pred_qvalues = self.qnet(states).gather(1, actions)
        qvalues = self.qnet(states)
        pred_qvalues = qvalues.gather(1, actions)
        #
        #policy_loss = -torch.sum(qvalues * torch.softmax(qvalues, dim=1), dim=1) + torch.logsumexp(qvalues, dim=1)
        #policy_loss = policy_loss.mean()
        #
        huberloss = huber(pred_qvalues, target_qvalues)
        loss = huberloss #+ 0.001 * policy_loss
        #print('==>', huberloss, policy_loss, '\n')
        loss.backward()
        self.optimizer.step()

    def learn(self, epochs, batch_size=64):
        for epoch in range(epochs):
            # sample new batch here
            batch, _ = self.memory.sample(batch_size)
            self.update_parameters(batch)
            soft_update(self._qnet, self.qnet, self.tau)

    def copy_weights(self):
        hard_update(self._qnet, self.qnet)

    def save(self):
        state = {
            'args': self.args,
            'qnet': self.qnet.state_dict(),
            '_qnet': self._qnet.state_dict(),
        }
        #'feature_dict': self.features.state_dict(),
        torch.save(state, self.file)
        print("Saved to " + self.file)

    def load(self, file='pretrained/dqn/dqn_model.pth.tar'):
        state = torch.load(file, map_location=lambda storage, loc: storage)
        if self.args != state['args']:
            print('Agent parameters from file are different from call')
            print('Overwriting agent to load file ... ')
            args = state['args']
            self.__init__(*args)

        self.qnet.load_state_dict(state['qnet'])
        self.copy_weights()
        print('Loaded')
        return

    def create_save_file(self):
        path = './pretrained/dqn'
        os.makedirs(path, exist_ok=True)
        self.file = next_path(path + '/' + 'dqn_model_%s.pth.tar')
