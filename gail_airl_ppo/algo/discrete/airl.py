import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import Adam
from tqdm import tqdm

from gail_airl_ppo.algo.discrete import AIRLDiscrim
from .ppo_discrete import PPO
torch.set_warn_always(False)

class AIRL(PPO):

    def __init__(self, buffer_exp, state_dim, action_dim, device, seed,
                gamma=0.995, rollout_length=10000, mix_buffer=1,
                batch_size=64, lr_actor=3e-4, lr_critic=3e-4, lr_disc=3e-4,
                units_disc_r=(100, 100), units_disc_v=(100, 100),
                epoch_ppo=50, epoch_disc=10, clip_eps=0.2, lambd=0.97,
                ent_coef=0.0, max_grad_norm=10.0, save_interval=1e5, 
                eval_interval=10000, num_eval_eps=100, net_width=64, 
                l2_reg=1e-3, adv_norm=False, ent_coef_decay=0.99):
        super().__init__(
            state_dim=state_dim,
            action_dim=action_dim,
            rollout_length=rollout_length,
            save_interval=save_interval,
            eval_interval=eval_interval,
            num_eval_episodes=num_eval_eps,
            gamma=gamma,
            lambd=lambd,
            net_width=net_width,
            lr_actor=lr_actor,
            lr_critic=lr_critic,
            clip_eps=clip_eps,
            epoch_ppo=epoch_ppo,
            batch_size=batch_size,
            l2_reg=l2_reg,
            entropy_coef=ent_coef,
            adv_normalization=adv_norm,
            entropy_coef_decay=ent_coef_decay,
            device=device,  
            mix_buffer=mix_buffer,
            )

        # Expert's buffer.
        self.buffer_exp = buffer_exp

        # Discriminator.
        self.disc = AIRLDiscrim(
            state_dim=state_dim,
            gamma=gamma,
            hidden_units_r=units_disc_r,
            hidden_units_v=units_disc_v,
            hidden_activation_r=nn.ReLU(inplace=True),
            hidden_activation_v=nn.ReLU(inplace=True)
        ).to(device)

        self.learning_steps_disc = 0
        self.learning_steps = 0
        self.optim_disc = Adam(self.disc.parameters(), lr=lr_disc)
        self.batch_size = batch_size
        self.epoch_disc = epoch_disc

    def update(self, writer):
        self.learning_steps += 1

        for _ in range(self.epoch_disc):
            self.learning_steps_disc += 1

            # Samples from current policy's trajectories.
            states, _, _, next_states, log_pis, _, dones  = \
                self.buffer.sample(self.batch_size)
            # Samples from expert's demonstrations.
            states_exp, actions_exp, _, next_states_exp, _, _, dones_exp = \
                self.buffer_exp.sample(self.batch_size)
            # Calculate log probabilities of expert actions.
            with torch.no_grad():
                log_pis_exp, _ = self.actor.evaluate(
                    states_exp, actions_exp)
            # Update discriminator.
            self.update_disc(
                states, dones, log_pis, next_states, states_exp,
                dones_exp, log_pis_exp, next_states_exp, writer
            )

        # We don't use reward signals here,
        states, actions, _, next_states, log_pis, state_values, dones = self.buffer.get()

        # Calculate rewards.
        rewards = self.disc.calculate_reward(
            states, dones, log_pis, next_states)

        # Update PPO using estimated rewards.
        self.update_ppo(
            states, actions, rewards, log_pis, state_values)

    def update_disc(self, states, dones, log_pis, next_states,
                    states_exp, dones_exp, log_pis_exp,
                    next_states_exp, writer):
        # Output of discriminator is (-inf, inf), not [0, 1].
        logits_pi = self.disc(states, dones, log_pis, next_states)
        logits_exp = self.disc(
            states_exp, dones_exp, log_pis_exp, next_states_exp)

        # Discriminator is to maximize E_{\pi} [log(1 - D)] + E_{exp} [log(D)].
        loss_pi = -F.logsigmoid(-logits_pi).mean()
        loss_exp = -F.logsigmoid(logits_exp).mean()
        loss_disc = loss_pi + loss_exp

        self.optim_disc.zero_grad()
        loss_disc.backward()
        self.optim_disc.step()

        if self.learning_steps_disc % self.epoch_disc == 0:
            writer.add_scalar(
                'loss/disc', loss_disc.item(), self.learning_steps)

            # Discriminator's accuracies.
            with torch.no_grad():
                acc_pi = (logits_pi < 0).float().mean().item()
                acc_exp = (logits_exp > 0).float().mean().item()
            print(f"Discriminator loss: {round(loss_disc.item(),4)}  | Accuracy exp: {round(acc_exp,4)}   | Accuracy pi: {round(acc_pi,4)} |")
            writer.add_scalar('stats/acc_pi', acc_pi, self.learning_steps)
            writer.add_scalar('stats/acc_exp', acc_exp, self.learning_steps)

    def evaluate(self, n_episode, tsteps, env, runtime, render=False):
        scores = 0
        scores_irl = 0
        rendering = False
        for ep in range(self.neval):
            s, done, ep_r, ep_r_irl, steps = env.reset(), False, 0, 0, 0
            s = torch.from_numpy(s).float().to(self.device)
            if ep < self.neval - 2: # Only render last episode
                rendering = render
            else: rendering = False
            while not done:
                # Take deterministic actions at test time
                a, _ = self.exploit(s)
                act = a.clone().cpu().numpy()
                s_prime, r, done, _ = env.step(act)
                log_pi,_ = self.actor_old.evaluate(s, a)
                s_prime = torch.from_numpy(s_prime).float().to(self.device)
                is_terminal = torch.as_tensor(done).float().to(self.device)
                r_irl = self.disc.calculate_reward(s, is_terminal, log_pi, s_prime)
                ep_r += r
                ep_r_irl += r_irl
                steps += 1
                s = s_prime
                if rendering:
                    env.render()
            scores += ep_r
            scores_irl += ep_r_irl.clone().cpu().numpy()[0] 
        print("-----------------------------------------------------------------------------------------------------------------------------")
        print(f'Epsiode: {n_episode:<6} |   '
            f'Num steps: {tsteps:<6}    |   '
            f'Return: {scores/self.neval:<5.1f} |   '
            f'IRL Return: {scores_irl/self.neval:<5.1f} |   '
            f'Elapsed Time: {runtime}   |')
        print("-----------------------------------------------------------------------------------------------------------------------------")
        return scores/self.neval, env