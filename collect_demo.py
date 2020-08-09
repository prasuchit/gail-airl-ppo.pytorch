import os
import argparse
import torch
import gym

from gail_ppo.algo import EXPERT_ALGOS
from gail_ppo.utils import collect_demo


def run(args):
    env = gym.make(args.env_id)

    algo = EXPERT_ALGOS[args.algo](
        state_shape=env.observation_space.shape,
        action_shape=env.action_space.shape,
        device=torch.device("cuda" if args.cuda else "cpu"),
        path=args.weight
    )

    collect_demo(
        env=env,
        algo=algo,
        buffer_size=args.buffer_size,
        device=torch.device("cuda" if args.cuda else "cpu"),
        std=args.std,
        p_rand=args.p_rand,
        seed=args.seed
    ).save(os.path.join(
        'buffers',
        f'{args.env_id}_std{args.std}_prand{args.p_rand}.pth'
    ))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--weight', type=str, required=True)
    p.add_argument('--env_id', type=str, default='HalfCheetahBulletEnv-v0')
    p.add_argument('--algo', type=str, default='sac')
    p.add_argument('--buffer_size', type=int, default=10**5)
    p.add_argument('--std', type=float, default=0.0)
    p.add_argument('--p_rand', type=float, default=0.0)
    p.add_argument('--cuda', action='store_true')
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args()
    run(args)