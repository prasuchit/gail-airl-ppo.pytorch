import os
from datetime import timedelta
from pathlib import Path
from time import time

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

PACKAGE_PATH = Path(__file__).parents[0]	# Abs path of package

class Trainer:

    def __init__(self, env_id, env, eval_env, algo, log_dir, seed=0, num_steps=10**5,
                write=False, render=False, save_path=f"{PACKAGE_PATH}/model"):
        super().__init__()

        # Env to collect samples.
        self.seed = seed
        
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        self.env_id = env_id
        self.env = env
        self.env.seed(seed)

        # Env for evaluation.
        self.eval_env = eval_env
        self.eval_env.seed(2**31-seed)

        self.algo = algo
        self.log_dir = log_dir
        self.save_path = save_path

        # Log setting.
        self.summary_dir = os.path.join(log_dir, 'summary')
        self.write = write        
        self.render = render
        self.writer = SummaryWriter(log_dir=self.summary_dir)
        # self.model_dir = os.path.join(log_dir, 'model')
        # if not os.path.exists(self.model_dir):
        #     os.makedirs(self.model_dir)

        # Other parameters.
        self.num_steps = num_steps
        
    def train(self):        
        # Time to start training.
        self.start_time = time()
        # # Initialize the environment.
        traj_lenth = 0
        total_steps = 0
        n_episode = 0
        # max_e_steps = self.env._max_episode_steps        
        while total_steps < self.num_steps:
            n_episode += 1
            s, done, steps, ep_r = self.env.reset(), False, 0, 0
            '''Interact & train'''
            while not done:
                total_steps += 1
                traj_lenth += 1
                steps += 1
                
                s_prime, a, pi_a, r, done, s_val, self.env = self.algo.step(self.env, s, self.render)

                # if (done and steps != max_e_steps):
                #     # if self.env_id == 'LunarLander-v2':
                #     #     if r <=-100: r = -30  # good for LunarLander
                #     dw = True  #dw: dead and win
                # else:
                #     dw = False

                self.algo.buffer.put((s, a, r, s_prime, pi_a, s_val, done))
                s = s_prime
                ep_r += r

                '''update if its time'''
                # if not self.render:
                if self.algo.is_update(total_steps):
                    total_loss = self.algo.update(self.writer)
                    traj_lenth = 0
                    if self.write:
                        self.writer.add_scalar('total_loss', total_loss, global_step=total_steps)

                '''record & log'''
                if self.algo.is_eval(total_steps):
                    render = False
                    if total_steps > self.num_steps - 100:
                        render = self.render
                    score, self.eval_env = self.algo.evaluate(n_episode = n_episode, tsteps=total_steps, env=self.eval_env, runtime=self.time, render=render)
                    if self.write:
                        self.writer.add_scalar('ep_r', score, global_step=total_steps)

                '''save model'''
                if self.algo.is_save(total_steps):
                    self.algo.save_models(step=total_steps, env_id=self.env_id, save_path=self.save_path, last_score=score)
                    
        self.env.close()
        self.eval_env.close()        
        
    def evaluate(self, step):
        # self.algo.evaluate()
        raise NotImplementedError

    @property
    def time(self):
        return str(timedelta(seconds=int(time() - self.start_time)))
