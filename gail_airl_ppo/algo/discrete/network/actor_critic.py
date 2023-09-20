
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, net_width):
        super(Actor, self).__init__()
        # actor
        self.actor = nn.Sequential(
                        nn.Linear(state_dim, net_width),
                        nn.Tanh(),
                        nn.Linear(net_width, net_width),
                        nn.Tanh(),
                        nn.Linear(net_width, action_dim),
                        nn.Softmax(dim=-1)
                    )
    
    def forward(self, state):
        # action_probs = self.actor(state)
        # dist = Categorical(action_probs)

        # action = dist.sample()
        # action_logprob = dist.log_prob(action)

        # return action, action_logprob
        raise NotImplementedError

    def act(self, state):
        action_probs = self.actor(state)
        dist = Categorical(action_probs)

        action = dist.sample()
        action_logprob = dist.log_prob(action)

        return action.detach(), action_logprob.detach()
    
    def exploit(self, state):
        
        action_probs = self.actor(state)
        action = torch.argmax(action_probs)
        
        return action.detach()
    
    def evaluate(self, state, action):

        action_probs = self.actor(state)
        dist = Categorical(action_probs)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        
        return action_logprobs, dist_entropy


class Critic(nn.Module):
    def __init__(self, state_dim, net_width):
        super(Critic, self).__init__()
        # critic
        self.critic = nn.Sequential(
                        nn.Linear(state_dim, net_width),
                        nn.Tanh(),
                        nn.Linear(net_width, net_width),
                        nn.Tanh(),
                        nn.Linear(net_width, 1)
                    )
        
    def forward(self, state):
        # state_val = self.critic(state)
        # return state_val
        raise NotImplementedError

    def react(self, state):
        state_val = self.critic(state)
        return state_val.detach()
    
    def evaluate(self, state):
        state_values = self.critic(state)
        return state_values