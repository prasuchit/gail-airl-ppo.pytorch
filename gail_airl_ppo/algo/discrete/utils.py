import torch
from torch import nn
import numpy as np
from tqdm import tqdm
from .rollout_buffer import RolloutBuffer as Buffer


def build_mlp(input_dim, output_dim, hidden_units=[64, 64],
            hidden_activation=nn.Tanh(), output_activation=None):
    layers = []
    units = input_dim
    for next_units in hidden_units:
        layers.append(nn.Linear(units, next_units))
        layers.append(hidden_activation)
        units = next_units
    layers.append(nn.Linear(units, output_dim))
    if output_activation is not None:
        layers.append(output_activation)
    return nn.Sequential(*layers)


def disable_gradient(network):
    for param in network.parameters():
        param.requires_grad = False


def collect_demo(env, algo, buffer_size, device, std, p_rand, seed=0):
    env.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    buffer = Buffer(
        buffer_size=buffer_size,
    )

    total_return = 0.0
    num_episodes = 0

    state = env.reset()
    t = 0
    episode_return = 0.0

    # for _ in tqdm(range(1, buffer_size + 1)):
    for _ in range(1, buffer_size + 1):
        t += 1
        
        state = torch.from_numpy(state).float().to(device)
        state_val = algo.critic_old.react(state)
        env.render()        
        if np.random.rand() < p_rand:
            action, log_probs = algo.explore(state)
        else:
            action, log_probs = algo.exploit(state)
            
        action = action.clone().cpu().numpy()

        next_state, reward, done, _ = env.step(action)
        mask = False if t == env._max_episode_steps else done
        buffer.put((state, action, reward, next_state, log_probs, state_val, mask))
        episode_return += reward

        if done:
            print("New Episode...")
            num_episodes += 1
            total_return += episode_return
            state = env.reset()
            t = 0
            episode_return = 0.0

        state = next_state

    print(f'Mean return of the expert is {total_return / num_episodes}')
    return buffer

