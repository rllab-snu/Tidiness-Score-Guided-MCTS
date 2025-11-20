from pathlib import Path

import os
import sys
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.models import resnet18
from torchvision import transforms

from src.iql import ImplicitQLearning
from src.policy import DiscreteResNetPolicy, DeterministicResNetPolicy, DiscreteTransportPolicy, DeterministicTransportPolicy, GaussianPolicy
from src.value_functions import TransportQ, ResNetTwinQ, ValueFunction, RewardFunction
from src.util import return_range, set_seed, Log, sample_batch, torchify, evaluate_policy
from src.util import DEFAULT_DEVICE
from iql_dataset import TabletopOfflineDataset

import datetime
import wandb
from matplotlib import pyplot as plt

file_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(file_path, '../policy_learning'))
import nonechucks as nc


def loadRewardFunction(model_path):
    gNet = resnet18(pretrained=False)
    fc_in_features = gNet.fc.in_features
    gNet.fc = nn.Sequential(
        nn.Linear(fc_in_features, 1),
    )
    gNet.load_state_dict(torch.load(model_path))
    gNet.to(DEFAULT_DEVICE)
    gNet.eval()
    preprocess = transforms.Compose([
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    return gNet, preprocess

def evaluateBatch(data, policy, gNet, preprocess, H, W, cropSize):
    images_after_pick = data['image_after_pick'].to(torch.float32).to(DEFAULT_DEVICE)
    patches = data['next_patch'].to(torch.float32).to(DEFAULT_DEVICE)
    obs = [None, images_after_pick, patches]
    probs = policy(obs)[1].cpu().detach().numpy()
    gt_actions = data['action'].cpu().detach().numpy()

    imageSize = np.array([360, 480])
    tableSize = np.array([H, W])
    ratio = imageSize // tableSize
    offset = (imageSize - ratio * tableSize + ratio)//2

    images = data['image_after_pick'].detach().numpy()
    patches = data['next_patch'].detach().numpy()
    next_images = data['next_image'].detach().numpy()
    images_after_action = []
    for b in range(len(images)):
        image = images[b]
        patch = patches[b]
        action_prob = probs[b]
        gt_action = gt_actions[b]

        mask = np.zeros([cropSize, cropSize])
        mask[patch.sum(2) > 0] = 1
        next_image = next_images[b]

        image_after_action = np.copy(image)
        index = np.argmax(action_prob)
        py, px = index // W, index % W
        ty, tx = np.array([py, px]) * ratio + offset
        yMin = int(ty - cropSize / 2)
        yMax = int(ty + cropSize / 2)
        xMin = int(tx - cropSize / 2)
        xMax = int(tx + cropSize / 2)
        image_after_action[
                max(0, yMin): min(imageSize[0], yMax),
                max(0, xMin): min(imageSize[1], xMax)
        ] += (patch * mask[:, :, None])[
                            max(0, -yMin): max(0, -yMin) + (min(imageSize[0], yMax) - max(0, yMin)),
                            max(0, -xMin): max(0, -xMin) + (min(imageSize[1], xMax) - max(0, xMin)),
                            ]
        images_after_action.append(image_after_action)
    images_after_action = np.concatenate(images_after_action, 0).reshape([-1, 360, 480, 3])
    
    s = preprocess(torch.Tensor(next_images).permute([0,3,1,2])).cuda()
    rewards = gNet(s).cpu().detach().numpy()

    s_prime = preprocess(torch.Tensor(images_after_action).permute([0,3,1,2])).cuda()
    rewards_prime = gNet(s_prime).cpu().detach().numpy()
    return (rewards_prime - rewards).mean()


def main(args, log_name):
    torch.set_num_threads(1)
    log = Log(Path(args.log_dir), vars(args))
    log(f'Log dir: {log.dir}')

    H, W = 12, 15
    train_dataset = TabletopOfflineDataset(args.data_dir, crop_size=args.crop_size, H=H, W=W, view='top', gaussian=args.gaussian)
    test_dataset = TabletopOfflineDataset(args.data_dir, crop_size=args.crop_size, H=H, W=W, view='top', gaussian=args.gaussian)
    indices_test = np.arange(len(test_dataset))[::4][-1000:].tolist()
    test_dataset.data_rewards = [test_dataset.data_rewards[d] for d in indices_test]
    test_dataset.data_terminals = [test_dataset.data_terminals[d] for d in indices_test]
    test_dataset.data_next_images = [test_dataset.data_next_images[d] for d in indices_test]
    test_dataset.data_images = [test_dataset.data_images[d] for d in indices_test]
    test_dataset.data_next_segs = [test_dataset.data_next_segs[d] for d in indices_test]
    test_dataset.data_segs = [test_dataset.data_segs[d] for d in indices_test]
    test_dataset.data_next_obj_infos = [test_dataset.data_next_obj_infos[d] for d in indices_test]
    test_dataset.data_obj_infos = [test_dataset.data_obj_infos[d] for d in indices_test]
    train_dataset, test_dataset = nc.SafeDataset(train_dataset), nc.SafeDataset(test_dataset)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # dataset = TabletopOfflineDataset(data_dir=args.data_dir, crop_size=args.crop_size, view='top')
    # dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    set_seed(args.seed)
    
    if args.continuous_policy:
        policy = GaussianPolicy()
    elif args.deterministic_policy:
        if args.policy_net=='transport':
            policy = DeterministicTransportPolicy(crop_size=args.crop_size)
        elif args.policy_net=='resnet':
            policy = DeterministicResNetPolicy(crop_size=args.crop_size)
            
    else:
        if args.policy_net=='transport':
            policy = DiscreteTransportPolicy(crop_size=args.crop_size)
        elif args.policy_net=='resnet':
            policy = DiscreteResNetPolicy(crop_size=args.crop_size)

    if args.q_net=='transport':
        qNet = TransportQ(crop_size=args.crop_size)
    else:
        qNet = ResNetTwinQ(crop_size=args.crop_size)
    vNet = ValueFunction(hidden_dim=args.hidden_dim)
    rNet = RewardFunction(hidden_dim=args.hidden_dim, sigmoid=args.sigmoid)
    iql = ImplicitQLearning(
        qf=qNet, #winQ(crop_size=args.crop_size),
        vf=vNet, #ValueFunction(hidden_dim=args.hidden_dim),
        rf=rNet, #ValueFunction(hidden_dim=args.hidden_dim),
        policy=policy,
        q_optimizer_factory=lambda params: torch.optim.Adam(params, lr=args.q_learning_rate, weight_decay=1e-5),
        v_optimizer_factory=lambda params: torch.optim.Adam(params, lr=args.v_learning_rate, weight_decay=1e-5),
        r_optimizer_factory=lambda params: torch.optim.Adam(params, lr=args.r_learning_rate, weight_decay=1e-5),
        policy_optimizer_factory=lambda params: torch.optim.Adam(params, lr=args.policy_learning_rate, weight_decay=1e-5),
        max_steps=args.n_epochs * len(train_loader) // args.batch_size,
        tau=args.tau,
        beta=args.beta,
        alpha=args.alpha,
        discount=args.discount
    )
    gNet, preprocess = loadRewardFunction(args.reward_model_path)

    torch.autograd.set_detect_anomaly(True)
    count_steps = 0
    for epoch in range(args.n_epochs):
        with tqdm(train_loader) as bar:
            bar.set_description(f'Epoch {epoch}')
            for batch in bar:
                images = batch['image'].to(torch.float32).to(DEFAULT_DEVICE)
                images_after_pick = batch['image_after_pick'].to(torch.float32).to(DEFAULT_DEVICE)
                patches = batch['next_patch'].to(torch.float32).to(DEFAULT_DEVICE)
                # action distribution
                if args.action_distribution:
                    action_labels = batch['action_dist'].to(torch.float32).to(DEFAULT_DEVICE)
                else:
                    actions = batch['action']
                    labels = np.zeros([len(actions), H, W])
                    labels[np.arange(len(actions)), actions[:, 0], actions[:, 1]] = 1
                    action_labels = torch.Tensor(labels).to(torch.float32).to(DEFAULT_DEVICE)

                next_images = batch['next_image'].to(torch.float32).to(DEFAULT_DEVICE)
                if args.reward=='classifier':
                    s = preprocess(images.permute([0,3,1,2]))
                    s_prime = preprocess(next_images.permute([0,3,1,2]))
                    states = torch.cat([s, s_prime], 0)
                    with torch.no_grad():
                        scores = gNet(states) #.cpu().detach().numpy()
                        rewards = scores[len(scores)//2:] - scores[:len(scores)//2]
                    rewards = rewards.view(-1)
                elif args.reward=='score':
                    rewards = batch['next_score'] - batch['score']
                    rewards = rewards.to(torch.float32).to(DEFAULT_DEVICE)
                else:
                    rewards = batch['reward'].to(torch.float32).to(DEFAULT_DEVICE)
                terminals = batch['terminal'].to(DEFAULT_DEVICE)
                observations = [images, images_after_pick, patches]
                next_observations = [next_images, None, None]
                if args.reward=='score':
                    scores = batch['next_score'].to(torch.float32).to(DEFAULT_DEVICE)
                    losses = iql.update(observations, action_labels, next_observations, rewards, terminals, scores)
                else:
                    losses = iql.update(observations, action_labels, next_observations, rewards, terminals)
                bar.set_postfix(losses)
                if not args.wandb_off:
                    wandb.log(losses, count_steps)
                count_steps += 1
                if count_steps%args.eval_period==0:
                    # Evaluate policy
                    delta_rewards = []
                    for t_data in test_loader:
                        delta_reward = evaluateBatch(t_data, iql.policy, gNet, preprocess, H, W, args.crop_size)
                        delta_rewards.append(delta_reward)
                    print('rewards:', np.mean(delta_rewards))
                    if not args.wandb_off:
                        wandb.log({'Delta reward': np.mean(delta_rewards)}, count_steps)
        if not os.path.isdir(os.path.join(args.log_dir, log_name)):
            os.makedirs(os.path.join(args.log_dir, log_name))
        torch.save(iql.state_dict(), os.path.join(args.log_dir, log_name, 'iql_e%d.pth'%(epoch+1)))

    torch.save(iql.state_dict(), os.path.join(args.log_dir, log_name, 'iql_final.pth'))
    #torch.save(iql.state_dict(), log.dir/'final.pt')
    log.close()


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--data-dir', type=str, default='/ssd/disk/TableTidyingUp/dataset_template/train')
    parser.add_argument('--log-dir', type=str, default='logs')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--discount', type=float, default=0.9)
    parser.add_argument('--crop-size', type=int, default=128) #96
    parser.add_argument('--hidden-dim', type=int, default=256)
    parser.add_argument('--n-epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--q-learning-rate', type=float, default=1e-4)
    parser.add_argument('--v-learning-rate', type=float, default=1e-4)
    parser.add_argument('--r-learning-rate', type=float, default=1e-4)
    parser.add_argument('--policy-learning-rate', type=float, default=1e-4)
    parser.add_argument('--alpha', type=float, default=0.005)
    parser.add_argument('--tau', type=float, default=0.7)
    parser.add_argument('--beta', type=float, default=3.0)
    parser.add_argument('--action-distribution', action='store_true')
    parser.add_argument('--continuous-policy', action='store_true')
    parser.add_argument('--deterministic-policy', action='store_true') # otherwise, use a categorical policy
    parser.add_argument('--q-net', type=str, default='resnet') # 'transport' / 'resnet
    parser.add_argument('--policy-net', type=str, default='resnet') # 'transport' / 'resnet
    parser.add_argument('--reward', type=str, default='') # '' / 'classifier' / 'score'
    parser.add_argument('--reward-model-path', type=str, default='../mcts/data/classification-best/top_nobg_linspace_mse-best.pth')
    parser.add_argument('--sigmoid', action='store_true')
    parser.add_argument('--eval-period', type=int, default=2000)
    parser.add_argument('--wandb-off', action='store_true')
    parser.add_argument('--gaussian', action='store_true')
    args = parser.parse_args()

    now = datetime.datetime.now()
    log_name = now.strftime("%m%d_%H%M")
    if not args.wandb_off:
        wandb.init(project="IQL")
        wandb.config.update(parser.parse_args())
        wandb.run.name = log_name
        wandb.run.save()

    main(args, log_name)
