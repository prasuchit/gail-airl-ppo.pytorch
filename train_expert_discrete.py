import argparse
import os
from datetime import datetime
from pathlib import Path
from time import time

import gym
import torch
import yaml

from gail_airl_ppo.algo.discrete import PPO
from gail_airl_ppo.trainer_discrete import Trainer


def run(args):
    env = gym.make(args.env_id)
    eval_env = gym.make(args.env_id)
    state_dim = env.observation_space.shape[0]
    action_dim = action_dim = env.action_space.n
    device = torch.device("cuda" if (torch.cuda.is_available() and args.cuda) else "cpu")
    torch.cuda.empty_cache()
    kwargs = {
        # "env_with_Dead": env_with_Dead[EnvIdex],
        "state_dim": state_dim,
        "action_dim": action_dim,
        "rollout_length": args.rollout_length,
        "save_interval": args.save_interval,
        "eval_interval": args.eval_interval,
        "num_eval_episodes": args.num_eval_eps,
        "gamma": args.gamma,
        "lambd": args.lambd,
        "net_width": args.net_width,
        "lr_actor": args.lr_actor,
        "lr_critic": args.lr_critic,
        "clip_eps": args.clip_eps,
        "epoch_ppo": args.epoch_ppo,
        "batch_size": args.batch_size,
        "l2_reg": args.l2_reg,
        "entropy_coef": args.ent_coef,  # hard env needs large value
        "adv_normalization": args.adv_norm,
        "entropy_coef_decay": args.ent_coef_decay,
        "device": device,  
    }

    algo = PPO(
        **kwargs
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
    )
    trainer.train()


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--num_steps', type=int, default=3e6)
    p.add_argument('--num_eval_eps', type=int, default=100)
    p.add_argument('--rollout_length', type=int, default=1000)
    p.add_argument('--eval_interval', type=int, default=10000)
    p.add_argument('--save_interval', type=int, default=1e5)
    p.add_argument('--net_width', type=int, default=64, help='Hidden net width')
    p.add_argument('--epoch_ppo', type=int, default=80, help='PPO update times')
    p.add_argument('--batch_size', type=int, default=64, help='lenth of sliced trajectory')
    p.add_argument('--adv_norm', type=bool, default=False)
    p.add_argument('--gamma', type=float, default=0.99)
    p.add_argument('--lambd', type=float, default=0.95)
    p.add_argument('--l2_reg', type=float, default=1e-3)
    p.add_argument('--ent_coef', type=float, default=1e-3)
    p.add_argument('--ent_coef_decay', type=float, default=0.99)
    p.add_argument('--clip_eps', type=float, default=0.2)
    p.add_argument('--lr_actor', type=float, default=0.0003)
    p.add_argument('--lr_critic', type=float, default=0.001)
    p.add_argument('--env_id', type=str, default="CartPole-v1")
    # p.add_argument('--env_id', type=str, default='LunarLander-v2')
    p.add_argument('--cuda', type=bool, default=True)
    # p.add_argument('--seed', type=int, default=209)
    # p.add_argument('--seed', type=int, default=int(time()))
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--render', type=bool, default=False)
    p.add_argument('--write', type=bool, default=False)
    args = p.parse_args()
    run(args)
