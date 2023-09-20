import numpy as np
import torch
import os

class SerializedBuffer:

    def __init__(self, path, device):
        tmp = torch.load(path)
        self.buffer_size = self._n = len(tmp['state'][0])
        self.device = device

        self.states = tmp['state'].clone().to(self.device)
        self.actions = tmp['action'].clone().to(self.device)
        self.rewards = tmp['reward'].clone().to(self.device)
        self.next_states = tmp['next_state'].clone().to(self.device)
        self.logprobs = tmp['logprobs'].clone().to(self.device)
        self.state_values = tmp['state_values'].clone().to(self.device)
        self.is_terminals = tmp['is_terminals'].clone().to(self.device)

    def sample(self, batch_size):
        # idxes = np.random.randint(low=0, high=self._n, size=batch_size)
        idxes = torch.randint(0, self._n, (batch_size,)).to(self.device)
        return (
            torch.index_select(self.states, 0, idxes),
            torch.index_select(self.actions, 0, idxes),
            torch.index_select(self.rewards, 0, idxes),
            torch.index_select(self.next_states, 0, idxes),
            torch.index_select(self.logprobs, 0, idxes),
            torch.index_select(self.state_values, 0, idxes),
            torch.index_select(self.is_terminals, 0, idxes),
        )


class RolloutBuffer:
    def __init__(self, buffer_size, mix=1, device='cpu'):
        self._n = 0
        self._p = 0
        self.mix = mix
        self.buffer_size = buffer_size
        self.total_size = mix * buffer_size
        self.device = device 
        self.actions = []
        self.states = []
        self.next_states = []
        self.logprobs = []
        self.rewards = []
        self.state_values = []
        self.is_terminals = []
    
    def clear(self):
        del self.states[:]
        del self.actions[:]
        del self.rewards[:]
        del self.next_states[:]
        del self.logprobs[:]
        del self.state_values[:]
        del self.is_terminals[:]
    
    def put(self, transition):
        state, action, reward, next_state, action_logprob, state_val, is_terminal = transition
        
        state = torch.as_tensor(state).float().to(self.device)
        action = torch.as_tensor(action).float().to(self.device)
        reward = torch.as_tensor(reward).float().to(self.device)
        next_state = torch.as_tensor(next_state).float().to(self.device)
        logprobs = torch.as_tensor(action_logprob).float().to(self.device)
        state_val = torch.as_tensor(state_val).float().to(self.device)
        is_terminal = torch.as_tensor(is_terminal).float().to(self.device)
        
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        self.logprobs.append(logprobs)
        self.state_values.append(state_val)
        self.is_terminals.append(is_terminal)
        
        self._p = (self._p + 1) % self.buffer_size  # Pointer to keep track of current index
        self._n = min(self._n + 1, self.buffer_size)    # Current size of buffer
        
    def get(self):
        assert self._p % self.buffer_size == 0
        start = (self._p - self.buffer_size) % self.total_size
        # idxes = slice(start, start + self.buffer_size)
        idxes = torch.arange(start, start + self.buffer_size).to(self.device) % self.total_size
        return (
            torch.index_select(torch.stack(self.states), 0, idxes),
            torch.index_select(torch.stack(self.actions), 0, idxes),
            torch.index_select(torch.stack(self.rewards), 0, idxes),
            torch.index_select(torch.stack(self.next_states), 0, idxes),
            torch.index_select(torch.stack(self.logprobs), 0, idxes),
            torch.index_select(torch.stack(self.state_values), 0, idxes),
            torch.index_select(torch.stack(self.is_terminals), 0, idxes),
        )
        
    def sample(self, batch_size):
        assert self._p % self.buffer_size == 0
        # idxes = np.random.randint(low=0, high=self._n, size=batch_size)
        idxes = torch.randint(0, self._n, (batch_size,)).to(self.device)
        return (torch.index_select(torch.stack(self.states), 0, idxes),
                torch.index_select(torch.stack(self.actions), 0, idxes),
                torch.index_select(torch.stack(self.rewards), 0, idxes),
                torch.index_select(torch.stack(self.next_states), 0, idxes),
                torch.index_select(torch.stack(self.logprobs), 0, idxes),
                torch.index_select(torch.stack(self.state_values), 0, idxes),
                torch.index_select(torch.stack(self.is_terminals), 0, idxes),
        )
    
    def save(self, path):
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))
        states = torch.stack(self.states).float().to(self.device)
        actions = torch.stack(self.actions).float().to(self.device)
        rewards = torch.stack(self.rewards).float().to(self.device)
        next_states = torch.stack(self.next_states).float().to(self.device)
        logprobs = torch.stack(self.logprobs).float().to(self.device)
        state_values = torch.stack(self.state_values).float().to(self.device)
        is_terminals = torch.stack(self.is_terminals).float().to(self.device)
        torch.save({
            'state': states,
            'action': actions,
            'reward': rewards,
            'next_state': next_states,
            'logprobs': logprobs,
            'state_values': state_values,
            'is_terminals': is_terminals,
        }, path)
