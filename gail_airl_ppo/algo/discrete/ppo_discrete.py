import copy
import math
from pathlib import Path
from time import sleep

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from tqdm import tqdm

from gail_airl_ppo.algo.base import Algorithm
from gail_airl_ppo.algo.discrete.network.actor_critic import Actor, Critic
from gail_airl_ppo.algo.discrete.rollout_buffer import RolloutBuffer
from gail_airl_ppo.algo.discrete.utils import disable_gradient

PACKAGE_PATH = Path(__file__).parents[2]	# Abs path of package

class PPO(Algorithm):
    
    def __init__(
            self,
            # env_with_Dead,
            state_dim,
            action_dim,
            gamma=0.99,
            rollout_length=2048, 
            lambd=0.95,
            net_width=200,
            lr_actor=1e-4,
            lr_critic=1e-4,
            clip_eps=0.2,
            epoch_ppo=10,
            batch_size=64,
            l2_reg=1e-3,
            entropy_coef = 1e-3,
            adv_normalization = False,
            entropy_coef_decay = 0.99,
            save_interval=10**3, 
            eval_interval=10**3, 
            num_eval_episodes=5,
            device="cpu",
            seed=0, 
            mix_buffer=1,
    ):

        self.device = device        
        self.actor = Actor(state_dim, action_dim, net_width).float().to(self.device)
        self.actor_old = Actor(state_dim, action_dim, net_width).float().to(self.device)

        self.critic = Critic(state_dim, net_width).float().to(self.device)
        self.critic_old = Critic(state_dim, net_width).float().to(self.device)
        
        self.optimizer = torch.optim.Adam([
                                {'params': self.actor.parameters(), 'lr': lr_actor},
                                {'params': self.critic.parameters(), 'lr': lr_critic}
                            ])
        
        self.MseLoss = nn.MSELoss()
        
        self.s_dim = state_dim
        self.data = []
        # self.env_with_Dead = env_with_Dead
        self.gamma = gamma
        self.lambd = lambd
        self.clip_eps = clip_eps
        self.epoch_ppo = epoch_ppo
        self.optim_batch_size = batch_size
        self.l2_reg = l2_reg
        self.entropy_coef = entropy_coef
        self.adv_normalization = adv_normalization
        self.entropy_coef_decay = entropy_coef_decay
        self.rollout_length = rollout_length    # Length of the longest trajectory        
        self.save_interval = save_interval
        self.eval_interval = eval_interval        
        self.neval = num_eval_episodes
        self.buffer = RolloutBuffer(buffer_size=rollout_length, mix=mix_buffer, device=self.device)
        self.learning_steps = 0
        self.learning_steps_ppo = 0
        self.max_grad_norm = 10

    def explore(self, state):
        '''Stochastic Policy'''
        if not torch.is_tensor(state):
            state = torch.as_tensor(state, dtype=torch.float).to(self.device)
        with torch.no_grad():
            action, action_logprob = self.actor_old.act(state)
        return action, action_logprob

    def exploit(self, state):
        '''Deterministic Policy'''
        if not torch.is_tensor(state):
            state = torch.as_tensor(state, dtype=torch.float).to(self.device)
        with torch.no_grad():
            action = self.actor_old.exploit(state)
        return action, 1.0
    
    def is_update(self, step):
        return step % (self.rollout_length * 4) == 0
    
    def is_eval(self, step):
        return step % self.eval_interval == 0
        
    def is_save(self, step):
        return step % self.save_interval == 0
    
    def step(self, env, s, render):
        
        s = torch.as_tensor(s).float().to(self.device)
        a, pi_a = self.explore(s)
        s_val = self.critic_old.react(s)
        a = a.clone().cpu().numpy()
        s_prime, r, done, info = env.step(a)
        
        return s_prime, a, pi_a, r, done, s_val, env
    
    def update(self, writer=None):
        # Monte Carlo estimate of returns
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
            
        # Normalizing the rewards
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)
        # convert list to tensor
        old_states = torch.squeeze(torch.stack(self.buffer.states, dim=0)).detach().to(self.device)
        old_actions = torch.squeeze(torch.stack(self.buffer.actions, dim=0)).detach().to(self.device)
        old_logprobs = torch.squeeze(torch.stack(self.buffer.logprobs, dim=0)).detach().to(self.device)
        old_state_values = torch.squeeze(torch.stack(self.buffer.state_values, dim=0)).detach().to(self.device)

        return self.update_ppo(old_states, old_actions, rewards, old_logprobs, old_state_values)

    def update_ppo(self, old_states, old_actions, rewards, old_logprobs, old_state_values):
        # calculate advantages
        advantages = rewards.detach() - old_state_values.detach()

        # Optimize policy for K epochs
        for _ in range(self.epoch_ppo):

            # Evaluating old actions and values
            logprobs, dist_entropy = self.actor.evaluate(old_states, old_actions)
            state_values = self.critic.evaluate(old_states)

            # match state_values tensor dimensions with rewards tensor
            state_values = torch.squeeze(state_values)
            
            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss  
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.clip_eps, 1+self.clip_eps) * advantages

            # final loss of clipped objective PPO
            loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, rewards) - 0.01 * dist_entropy
            
            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
            nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
            self.optimizer.step()
            
        # Copy new weights into old policy
        self.actor_old.load_state_dict(self.actor.state_dict())
        self.critic_old.load_state_dict(self.critic.state_dict())

        # clear buffer
        self.buffer.clear()
        
        return loss.mean()
        
    def evaluate(self, n_episode, tsteps, env, runtime, render=False):
        scores = 0
        for _ in range(self.neval):
            s, done, ep_r, steps = env.reset(), False, 0, 0
            while not done:
                # Take deterministic actions at test time
                a, _ = self.exploit(torch.from_numpy(s).float().to(self.device))
                a = a.clone().cpu().numpy()
                s_prime, r, done, info = env.step(a)
                ep_r += r
                steps += 1
                s = s_prime
                if render:
                    env.render()
            scores += ep_r   
        print("--------------------------------------------------------------------------------------------")
        print(f'Epsiode: {n_episode:<6} |   '
            f'Num steps: {tsteps:<6}    |   '
            f'Return: {scores/self.neval:<5.1f} |   '
            f'Elapsed Time: {runtime}')
        print("--------------------------------------------------------------------------------------------")
        return scores/self.neval, env

    def save_models(self, step, env_id, save_path=f"{PACKAGE_PATH}/model", last_score=0):
        critic_save_path = f"{save_path}/{env_id}/critic"
        actor_save_path = f"{save_path}/{env_id}/actor"
        Path(critic_save_path).mkdir(parents=True, exist_ok=True)
        Path(actor_save_path).mkdir(parents=True, exist_ok=True)
        torch.save(self.critic_old.state_dict(), f"{critic_save_path}/ppo_critic_step{step}_rew{last_score}.pth")
        torch.save(self.actor_old.state_dict(), f"{actor_save_path}/ppo_actor_step{step}_rew{last_score}.pth")

    def load(self, step, env_id, load_path=f"{PACKAGE_PATH}/model",last_score=0):
        critic_load_path = f"{load_path}/{env_id}/critic/ppo_critic_step{step}_rew{last_score}.pth"
        actor_load_path = f"{load_path}/{env_id}/actor/ppo_actor_step{step}_rew{last_score}.pth"
        self.critic_old.load_state_dict(torch.load(critic_load_path, map_location=lambda storage, loc: storage))
        self.actor_old.load_state_dict(torch.load(actor_load_path, map_location=lambda storage, loc: storage))


class PPOExpert(PPO):

    def __init__(self, state_dim, action_dim, net_width, device, actor_path, critic_path):
        self.actor_old = Actor(state_dim, action_dim, net_width).float().to(device)
        self.actor_old.load_state_dict(torch.load(actor_path))
        self.critic_old = Critic(state_dim=state_dim, net_width=net_width).float().to(device)
        self.critic_old.load_state_dict(torch.load(critic_path))
        disable_gradient(self.actor_old)
        self.device = device
