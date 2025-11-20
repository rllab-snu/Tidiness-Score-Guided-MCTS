import os
import sys
import numpy as np
import datetime
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from matplotlib import pyplot as plt
from argparse import ArgumentParser
from torchvision import transforms
from torchvision.models import resnet18
from src.policy import DiscreteResNetPolicy, DeterministicResNetPolicy, DiscreteTransportPolicy, DeterministicTransportPolicy, GaussianPolicy

file_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(file_path, '../policy_learning'))
import nonechucks as nc
from custom_dataset import TabletopOfflineDataset

Device = 'cuda'

def loadRewardFunction(model_path):
    vNet = resnet18(pretrained=False)
    fc_in_features = vNet.fc.in_features
    vNet.fc = nn.Sequential(
        nn.Linear(fc_in_features, 1),
    )
    vNet.load_state_dict(torch.load(model_path))
    vNet.to(Device)
    vNet.eval()
    preprocess = transforms.Compose([
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    return vNet, preprocess

def evaluateBatch(data, pnet, vNet, preprocess, H, W, cropSize):
    inputs = preprocess(data['image_after_pick'].permute([0,3,1,2])).to(torch.float32).to(Device)
    probs = pnet(inputs).cpu().detach().numpy()
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
        if False:
            plt.subplot(2, 3, 1)
            plt.imshow(patch)
            plt.subplot(2, 3, 2)
            plt.imshow(action_prob)
            plt.subplot(2, 3, 3)
            gt_action_prob = np.zeros_like(action_prob)
            gt_action_prob[gt_action[0], gt_action[1]] = 1
            plt.imshow(gt_action_prob)
            #plt.imshow(action_prob==action_prob.max())
            plt.subplot(2, 3, 4)
            plt.imshow(image)
            plt.subplot(2, 3, 5)
            plt.imshow(image_after_action)
            plt.subplot(2, 3, 6)
            plt.imshow(next_image)
            plt.show()
    images_after_action = np.concatenate(images_after_action, 0).reshape([-1, 360, 480, 3])
    
    s = preprocess(torch.Tensor(next_images).permute([0,3,1,2])).cuda()
    rewards = vNet(s).cpu().detach().numpy()

    s_prime = preprocess(torch.Tensor(images_after_action).permute([0,3,1,2])).cuda()
    rewards_prime = vNet(s_prime).cpu().detach().numpy()
    return (rewards_prime - rewards).mean()


def evaluate(args, log_name):
    H, W = 12, 15
    train_dataset = TabletopOfflineDataset(args.data_dir, crop_size=args.crop_size, H=H, W=W)
    test_dataset = TabletopOfflineDataset(args.data_dir, crop_size=args.crop_size, H=H, W=W)
    # train_size = int(0.9 * len(dataset))
    # test_size = len(dataset) - train_size
    # train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    
    indices_test = np.arange(len(test_dataset))[::8][-1000:].tolist()
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
    print("Train data:", len(train_dataset))
    print("Test data:", len(test_dataset))
    print("-"*60)

    if args.deterministic_policy:
        if args.policy_net=='transport':
            policy = DeterministicTransportPolicy(crop_size=args.crop_size)
        elif args.policy_net=='resnet':
            policy = DeterministicResNetPolicy(crop_size=args.crop_size)
            
    else:
        if args.policy_net=='transport':
            policy = DiscreteTransportPolicy(crop_size=args.crop_size)
        elif args.policy_net=='resnet':
            policy = DiscreteResNetPolicy(crop_size=args.crop_size)
    policy = policy.to(Device)
    state_dict = torch.load(args.model_path)
    state_dict = {k.replace('policy.', ''): v for k, v in state_dict.items() if k.startswith('policy.')}
    policy.load_state_dict(state_dict)

    vNet, preprocess = loadRewardFunction(args.reward_model_path)

    count_steps = 0
    with tqdm(test_loader) as bar:
        for i, data in enumerate(bar, 0):
            rgb = data['image_after_pick']
            if args.action_distribution:
                labels = data['action_dist'].to(torch.float32).to(Device)
            else:
                actions = data['action']
                labels = np.zeros([len(actions), H, W])
                labels[np.arange(len(actions)), actions[:, 0], actions[:, 1]] = 1
                labels = torch.Tensor(labels).to(torch.float32).to(Device)
            
            images_after_pick = rgb.to(torch.float32).to(Device)
            patches = data['next_patch'].to(torch.float32).to(Device)
            obs = [None, images_after_pick, patches]
            actions_predicted, probs, log_probs = policy(obs)
            
            if not os.path.isdir(os.path.join(args.log_dir, log_name)):
                os.makedirs(os.path.join(args.log_dir, log_name))
            for b in range(args.batch_size):
                fig = plt.figure(figsize=(16, 8))
                plt.subplot(1, 3, 1)
                plt.imshow(rgb.cpu().detach().numpy()[b])
                plt.subplot(1, 3, 2)
                p = probs.cpu().detach().numpy()
                plt.imshow(p[b], vmin=0., vmax=1.)
                texts = []
                for _y in range(p[b].shape[0]):
                    for _x in range(p[b].shape[1]):
                        if p[b][_y, _x] > 0.001:
                            txt = plt.text(_x, _y, '%.2f'%p[b][_y, _x], ha='center', va='center')
                        else:
                            txt = plt.text(_x, _y, '%.1f'%p[b][_y, _x], ha='center', va='center')
                        texts.append(txt)
                plt.subplot(1, 3, 3)
                plt.imshow(labels.cpu().detach().numpy()[b], vmin=0., vmax=1.)
                plt.savefig(os.path.join(args.log_dir, log_name, 'p-%d.png'%count_steps))
                for txt in texts:
                    txt.remove()
                #plt.savefig('../mcts/data/weekly/s-%d.png'%count_steps)
                count_steps += 1
                
            if False:
                delta_rewards = []
                for t_data in test_loader:
                    delta_reward = evaluateBatch(t_data, pnet, vNet, preprocess, H, W, args.crop_size)
                    delta_rewards.append(delta_reward)
                print('rewards:', np.mean(delta_rewards))
            

if __name__=='__main__':
    parser = ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--crop-size', type=int, default=96)
    parser.add_argument('--policy-net', type=str, default='resnet') # 'transport' / 'resnet
    parser.add_argument('--deterministic-policy', action='store_true') # otherwise, use a categorical policy
    parser.add_argument('--model-path', type=str, default='logs/0229_1619/iql_final.pth')
    parser.add_argument('--reward-model-path', type=str, default='../mcts/data/classification-best/top_nobg_linspace_mse-best.pth')
    parser.add_argument('--data-dir', type=str, default='/ssd/disk/TableTidyingUp/dataset_template/train')
    parser.add_argument('--log-dir', type=str, default='test')
    parser.add_argument('--action-distribution', action='store_true')
    args = parser.parse_args()

    now = datetime.datetime.now()
    log_name = now.strftime("%m%d_%H%M")
    
    evaluate(args, log_name)
    print('Finished Evluation')
