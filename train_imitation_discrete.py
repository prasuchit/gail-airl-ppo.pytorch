import argparse
import os
from datetime import datetime
from pathlib import Path

import gym
import torch

from gail_airl_ppo.algo.discrete import AIRL, SerializedBuffer
from gail_airl_ppo.trainer_discrete import Trainer

PACKAGE_PATH = Path(__file__).parents[0]	# Abs path of package

def run(args):
    env = gym.make(args.env_id)
    eval_env = gym.make(args.env_id)
    state_dim = env.observation_space.shape[0]
    action_dim = action_dim = env.action_space.n
    device = torch.device("cuda" if (torch.cuda.is_available() and args.cuda) else "cpu")
    torch.cuda.empty_cache()
    buffer_exp = SerializedBuffer(
        path=args.buffer,
        device=device
    )

    algo = AIRL(
        buffer_exp=buffer_exp,
        state_dim=state_dim,
        action_dim=action_dim,
        device=device,
        seed=args.seed,
        rollout_length=args.rollout_length,
        gamma=0.995, mix_buffer=1,
        batch_size=64, lr_actor=3e-4, lr_critic=3e-4, lr_disc=3e-4,
        units_disc_r=(100, 100), units_disc_v=(100, 100),
        epoch_ppo=args.epoch_ppo, epoch_disc=args.epoch_disc, clip_eps=0.2, lambd=0.97,
        ent_coef=1e-3, max_grad_norm=10.0, save_interval=args.save_interval, 
        eval_interval=args.eval_interval, num_eval_eps=100, net_width=64, 
        l2_reg=1e-3, adv_norm=False, ent_coef_decay=0.99,
    )

    time = datetime.now().strftime("%Y%m%d-%H%M")
    log_dir = os.path.join(
        'logs', args.env_id, 'ppo', f'seed{args.seed}-{time}')

    trainer = Trainer(
        env_id=args.env_id,
        env=env,
        eval_env=eval_env,
        algo=algo,
        log_dir=log_dir,
        num_steps=args.num_steps,
        seed=args.seed,
        render=args.render,
        write=args.write,
        save_path=args.save_path,
    )
    trainer.train()


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--buffer', type=str, default=f'{PACKAGE_PATH}/buffers/CartPole-v1/size1000000_std0.0_prand0.0.pth')
    p.add_argument('--save_path', type=str, default=f'{PACKAGE_PATH}/irl_models/rl_models')
    p.add_argument('--rollout_length', type=int, default=1000)
    p.add_argument('--num_steps', type=int, default=10**7)
    p.add_argument('--eval_interval', type=int, default=10000)
    p.add_argument('--save_interval', type=int, default=1e5)
    p.add_argument('--epoch_disc', type=int, default=20)
    p.add_argument('--epoch_ppo', type=int, default=200)
    # p.add_argument('--env_id', type=str, default='LunarLander-v2')
    p.add_argument('--env_id', type=str, default="CartPole-v1")
    p.add_argument('--algo', type=str, default='gail')
    p.add_argument('--cuda', type=bool, default=True)
    p.add_argument('--render', type=bool, default=True)
    p.add_argument('--write', type=bool, default=False)
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args()
    run(args)
