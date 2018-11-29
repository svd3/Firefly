import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

from .nets import Actor, Critic
from .utils import *
from .ReplayMemory import ReplayMemory

CUDA = torch.cuda.is_available()


class Agent():
    def __init__(self, input_dim, action_dim, hidden_dim=128, gamma=0.99, tau=0.001, memory_size=1e6):
        self.input_dim = input_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.gamma = gamma
        self.tau = tau
        print("Running DDPG Agent")
        if CUDA:
            self.actor = Actor(input_dim, action_dim, hidden_dim).cuda()
            self._actor = Actor(input_dim, action_dim, hidden_dim).cuda()
            self.critic = Critic(input_dim, action_dim, hidden_dim).cuda()
            self._critic = Critic(input_dim, action_dim, hidden_dim).cuda()
        else:
            self.actor = Actor(input_dim, action_dim, hidden_dim)
            self._actor = Actor(input_dim, action_dim, hidden_dim)
            self.critic = Critic(input_dim, action_dim, hidden_dim)
            self._critic = Critic(input_dim, action_dim, hidden_dim)

        self.actor_optim = Adam(self.actor.parameters(), lr=1e-4)
        self.critic_optim = Adam(self.critic.parameters(), lr=1e-3)

        self.priority = False
        self.memory = ReplayMemory(int(memory_size), priority=self.priority)

        self.args = (input_dim, action_dim, hidden_dim)
        hard_update(self._actor, self.actor)  # Make sure target is with the same weight
        hard_update(self._critic, self.critic)
        self.create_save_file()

    def select_action(self,  state, exploration=None):
        mu = self.actor(state).detach()
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
        masks = variable(torch.cat(batch.mask))
        with torch.no_grad():
            next_actions = self._actor(next_states)
            next_qvalues =  self._critic(next_states, next_actions)
            target_qvalues = rewards + self.gamma * next_qvalues

        self.critic_optim.zero_grad()
        pred_qvalues = self.critic(states, actions)
        #value_loss = F.mse_loss(pred_qvalues, target_qvalues)
        value_loss = torch.mean((pred_qvalues - target_qvalues)**2)
        value_loss.backward()
        self.critic_optim.step()

        self.actor_optim.zero_grad()
        policy_loss = -self.critic(states, self.actor(states))
        policy_loss = policy_loss.mean()
        policy_loss.backward()
        self.actor_optim.step()
        return policy_loss, value_loss

    def learn(self, epochs, batch_size=64):
        for epoch in range(epochs):
            # sample new batch here
            batch, _ = self.memory.sample(batch_size)
            losses = self.update_parameters(batch)
            soft_update(self._actor, self.actor, self.tau)
            soft_update(self._critic, self.critic, self.tau)

        return losses

    def save(self):
        state = {
            'args': self.args,
            'actor_dict': self.actor.state_dict(),
            'critic_dict': self.critic.state_dict(),
        }
        #'feature_dict': self.features.state_dict(),
        torch.save(state, self.file)
        print("Saved to " + self.file)

    def load(self, file='pretrained/ddpg/ddpg_model.pth.tar'):
        state = torch.load(file, map_location=lambda storage, loc: storage)
        if self.args != state['args']:
            print('Agent parameters from file are different from call')
            print('Overwriting agent to load file ... ')
            args = state['args']
            #self = Agent(*args)
            self.__init__(*args)

        self.actor.load_state_dict(state['actor_dict'])
        self.critic.load_state_dict(state['critic_dict'])
        hard_update(self._actor, self.actor)  # Make sure target is with the same weight
        hard_update(self._critic, self.critic)
        print('Loaded')
        return

    def create_save_file(self):
        #path = './pretrained/ddpg'
        path = './pretrained/ddpg_new'
        #path = './pretrained/ddpg_circle'
        os.makedirs(path, exist_ok=True)
        self.file = next_path(path + '/' + 'ddpg_model_%s.pth.tar')
        #self.file = next_path(path + '/' + 'circle_model_%s.pth.tar')
