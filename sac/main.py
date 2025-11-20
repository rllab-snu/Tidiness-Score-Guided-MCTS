import argparse
import datetime
import gym
import numpy as np
import itertools
import torch
import wandb
from sac import SAC
from torch.utils.tensorboard import SummaryWriter
from replay_memory import ReplayMemory

import os, sys
FILE_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(FILE_PATH, '..', 'mcts'))
from rl_env import Environment

import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='PyTorch Soft Actor-Critic Args')
# environment args #
parser.add_argument('--H', type=int, default=12)
parser.add_argument('--W', type=int, default=15)
parser.add_argument('--crop-size', type=int, default=128)
parser.add_argument('--num-objects', type=int, default=5)
parser.add_argument('--real', action="store_true")
parser.add_argument('--use-template', action="store_true")
parser.add_argument('--data-dir', type=str, default='/ssd/disk')
parser.add_argument('--gui-off', action="store_true")
parser.add_argument('--sim', action="store_true")
parser.add_argument('--dataset_dir', type=str, default='/ssd/disk/TableTidyingUp/dataset_template/train')
parser.add_argument('--max-length', type=int, default=20)
parser.add_argument('--threshold-success', type=float, default=0.9)
parser.add_argument('--reward-model-path', type=str, default='../mcts/data/classification-best/top_nobg_linspace_mse-best.pth')
parser.add_argument('--reward-type', type=str, default='delta-reward') # 'delta-rewrad' / 'binary'
parser.add_argument('--batch-size', type=int, default=32)
parser.add_argument('--wandb-off', action='store_true')
parser.add_argument('--use-policy', action='store_true')
parser.add_argument('--use-qnet', action='store_true')
parser.add_argument('--iql-path', type=str, default='../iql/logs/0308_0121/iql_e1.pth')
parser.add_argument('--policy-version', type=int, default=-1)
# SAC args #
parser.add_argument('--eval', type=bool, default=True,
                        help='Evaluates a policy a policy every 10 episode (default: True)')
parser.add_argument('--gamma', type=float, default=0.9, metavar='G', # 0.99
                        help='discount factor for reward (default: 0.99)')
parser.add_argument('--tau', type=float, default=0.005, metavar='G',
                        help='target smoothing coefficient(τ) (default: 0.005)')
parser.add_argument('--lr', type=float, default=0.0003, metavar='G',
                        help='learning rate (default: 0.0003)')
parser.add_argument('--alpha', type=float, default=0.2, metavar='G',
                        help='Temperature parameter α determines the relative importance of the entropy\
                            term against the reward (default: 0.2)')
parser.add_argument('--automatic_entropy_tuning', type=bool, default=False, metavar='G',
                        help='Automaically adjust α (default: False)')
parser.add_argument('--seed', type=int, default=123456, metavar='N',
                        help='random seed (default: 123456)')
parser.add_argument('--num_steps', type=int, default=1000001, metavar='N', # 1000001
                        help='maximum number of steps (default: 1000000)')
parser.add_argument('--hidden_size', type=int, default=256, metavar='N',
                        help='hidden size (default: 256)')
parser.add_argument('--updates_per_step', type=int, default=1, metavar='N',
                        help='model updates per simulator step (default: 1)')
parser.add_argument('--start_steps', type=int, default=2000, metavar='N', # 10000
                        help='Steps sampling random actions (default: 10000)')
parser.add_argument('--target_update_interval', type=int, default=1, metavar='N',
                        help='Value target update per no. of updates per step (default: 1)')
parser.add_argument('--replay_size', type=int, default=10000, metavar='N', # 10000000
                        help='size of replay buffer (default: 10000000)')
parser.add_argument('--save_freq', type=int, default=1000)
parser.add_argument('--continuous-policy', action="store_true")
#parser.add_argument('--cuda', action="store_true", help='run on CUDA (default: False)')
args = parser.parse_args()

# Environment
# env = NormalizedActions(gym.make(args.env_name))
env = Environment(args)
env.seed(args.seed)

# WanDB
now = datetime.datetime.now()
log_name = now.strftime("%m%d_%H%M")
if not args.wandb_off:
    wandb.init(project="SAC")
    wandb.config.update(parser.parse_args())
    wandb.run.name = log_name
    wandb.run.save()

# Agent
agent = SAC(args)

# Load pre-trained models
if args.use_policy:
    agent.load_policy(args.iql_path)
if args.use_qnet:
    agent.load_qnet(args.iql_path)

#Tesnorboard
writer = SummaryWriter('runs/{}_SAC'.format(log_name))

# Memory
memory = ReplayMemory(args.replay_size, args.seed)

# Training Loop
total_numsteps = 0
updates = 0

for i_episode in itertools.count(1):
    episode_reward = 0
    episode_steps = 0
    done = False
    obs = env.reset()

    while not done:
        if args.start_steps > total_numsteps:
            if np.random.random()<0.5:
                n = np.random.choice(np.arange(1, args.num_objects+1))
                y = np.random.choice(args.H)
                x = np.random.choice(args.W)
                rot = np.random.choice([1, 2])
                action = (n, y, x, rot)
            else:
                action = agent.select_action(obs)
        else:
            # n = np.random.choice(np.arange(1, args.num_objects+1))
            # rot = np.random.choice([1, 2])
            # rgb, rgbWoTargets, objectPatches = obs
            # state = [rgb, rgbWoTargets[n-1], objectPatches[n-1 + (rot-1)*args.num_objects]]
            action = agent.select_action(obs)  # Sample action from policy

        if len(memory) > args.batch_size:
            # Number of updates per step in environment
            for i in range(args.updates_per_step):
                # Update parameters of all the networks
                critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha = agent.update_parameters(memory, args.batch_size, updates)

                if not args.wandb_off:
                    logs = {'loss/critic_1': critic_1_loss,
                            'loss/critic_2': critic_2_loss,
                            'loss/policy': policy_loss,
                            'loss/entropy_loss': ent_loss,
                            'entropy_temprature/alpha': alpha}
                    wandb.log(logs, updates)
                writer.add_scalar('loss/critic_1', critic_1_loss, updates)
                writer.add_scalar('loss/critic_2', critic_2_loss, updates)
                writer.add_scalar('loss/policy', policy_loss, updates)
                writer.add_scalar('loss/entropy_loss', ent_loss, updates)
                writer.add_scalar('entropy_temprature/alpha', alpha, updates)
                updates += 1

        next_obs, reward, success, done = env.step(action) # Step
        episode_steps += 1
        total_numsteps += 1
        episode_reward += reward

        # Ignore the "done" signal if it comes from hitting the time horizon.
        # (https://github.com/openai/spinningup/blob/master/spinup/algos/sac/sac.py)
        mask = 1 if episode_steps == env.maxLength else float(not done)

        assert len(obs[1])==args.num_objects
        memory.push(obs, action, reward, next_obs, mask) # Append transition to memory

        obs = next_obs

    if total_numsteps > args.num_steps:
        break
    
    if not args.wandb_off:
        logs = {'reward/train': episode_reward}
        wandb.log(logs)#, i_episode)
    writer.add_scalar('reward/train', episode_reward, i_episode)
    print("Episode: {}, total numsteps: {}, episode steps: {}, reward: {}".format(i_episode, total_numsteps, episode_steps, round(episode_reward, 2)))

    if i_episode % args.save_freq==0:
        agent.save_checkpoint(log_name, str(i_episode))

    if i_episode % 100 == 0 and args.eval is True:
        avg_reward = 0.
        episodes = 10
        for _  in range(episodes):
            obs = env.reset()
            episode_reward = 0
            done = False
            while not done:
                action = agent.select_action(obs)

                next_obs, reward, success, done = env.step(action)
                episode_reward += reward

                obs = next_obs
            avg_reward += episode_reward
        avg_reward /= episodes

        if not args.wandb_off:
            logs = {'avg_reward/test': avg_reward}
            wandb.log(logs)#, i_episode)
        writer.add_scalar('avg_reward/test', avg_reward, i_episode)

        print("----------------------------------------")
        print("Test Episodes: {}, Avg. Reward: {}".format(episodes, round(avg_reward, 2)))
        print("----------------------------------------")

env.close()

