import os
import copy
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
from sac_utils import soft_update, hard_update
# from model import GaussianPolicy, QNetwork, DeterministicPolicy

import sys
FILE_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(FILE_PATH, '..', 'iql'))
from src.policy import DiscreteResNetPolicy, GaussianPolicy #PickPolicy
from src.policy import PolicyOpt0, PolicyOpt1, PolicyOpt2, PolicyOpt3, PolicyOpt4
from src.value_functions import ResNetTwinQ

import time

class SAC(object):
    def __init__(self, args):

        self.gamma = args.gamma
        self.tau = args.tau
        self.alpha = args.alpha

        self.target_update_interval = args.target_update_interval
        self.automatic_entropy_tuning = args.automatic_entropy_tuning

        self.device = torch.device("cuda") # if args.cuda else "cpu")

        self.critic = ResNetTwinQ(crop_size=args.crop_size).to(self.device)
        self.critic_optim = Adam(self.critic.parameters(), lr=args.lr)#, weight_decay=1e-5)
        self.critic_target = ResNetTwinQ(crop_size=args.crop_size).to(self.device)
        hard_update(self.critic_target, self.critic)
        
        # self.policy_pick = PickPolicy(crop_size=args.crop_size).to(self.device)
        self.policy_version = args.policy_version
        self.continuous_policy = args.continuous_policy
        
        if self.policy_version!=-1:
            if args.policy_version==0:
                self.policy_place = PolicyOpt0().to(self.device)
            elif args.policy_version==1:
                self.policy_place = PolicyOpt1().to(self.device)
            elif args.policy_version==2:
                self.policy_place = PolicyOpt2().to(self.device)
            elif args.policy_version==3:
                self.policy_place = PolicyOpt3().to(self.device)
            elif args.policy_version==4:
                self.policy_place = PolicyOpt4().to(self.device)
        elif self.continuous_policy:
            self.policy_place = GaussianPolicy().to(self.device)
        else:
            self.policy_place = DiscreteResNetPolicy(crop_size=args.crop_size).to(self.device)
        # self.policy_pick_optim = Adam(self.policy_pick.parameters(), lr=args.lr)#, weight_decay=1e-5)
        self.policy_place_optim = Adam(self.policy_place.parameters(), lr=args.lr)#, weight_decay=1e-5)

    def load_policy(self, path):
        state_dict = torch.load(path)
        state_dict = {k.replace('policy.', ''): v for k, v in state_dict.items() if k.startswith('policy.')}
        self.policy_place.load_state_dict(state_dict)

    def load_qnet(self, path):
        state_dict = torch.load(path)
        state_dict = {k.replace('qf.', ''): v for k, v in state_dict.items() if k.startswith('qf.')}
        self.critic.load_state_dict(state_dict)
        hard_update(self.critic_target, self.critic)
        
    def policy_sample(self, obs_batch):
        # st = time.time()
        actions = []
        log_pi = []

        pick = []
        rotation = []
        state_q = []
        patch = []
        for obs in obs_batch:
            rgb, rgbWoTargets, objectPatches = obs
            NB = len(rgbWoTargets)
            rgb = torch.FloatTensor(rgb).to(self.device)
            
            action_pick = np.random.choice(NB)  # random pick
            rot = np.random.choice(2)   # random rotation
            # obs = [rgb, rgbWoTargets, None]
            # action_pick, pick_probs, log_pick_probs, log_p_pick = self.policy_pick(obs)

            rgbWoTarget = rgbWoTargets[action_pick].copy()
            objectPatch = objectPatches[action_pick + rot*NB].copy()

            pick.append(action_pick+1)
            rotation.append(rot+1)
            state_q.append(rgbWoTarget[None, :])
            patch.append(objectPatch[None, :])
        
        state_q = torch.FloatTensor(np.concatenate(state_q, axis=0)).to(self.device)
        patch = torch.FloatTensor(np.concatenate(patch, axis=0)).to(self.device)
        policy_out = self.policy_place([None, state_q, patch])
        if self.continuous_policy:
            action_place = policy_out.sample()
            action_place = (action_place + 0.5) / torch.Tensor([12, 15]).cuda()
            action_place = torch.clamp(action_place, min=0, max=1-1e-8)
            action_place = action_place * torch.Tensor([12, 15]).cuda() - 0.5
            # action_place = torch.clamp(action_place / torch.Tensor([12, 15]).cuda(), min=0, max=1) * torch.Tensor([12, 15]).cuda()
            log_p_place = policy_out.log_prob(action_place)
        else:
            action_place, place_probs, log_place_probs, log_p_place = policy_out

        pick = np.array(pick)[:, None].astype(int)
        rotation = np.array(rotation)[:, None].astype(int)
        place = action_place.detach().cpu().numpy().round().astype(int)
        assert (place[:, 0]<12).all() and (place[:, 1]<15).all()
        actions = np.concatenate([pick, place, rotation], axis=1)
        # et = time.time()
        # print(et-st, 'secs.')
        # print()
        return actions, log_p_place
    
    def select_action(self, obs):
        actions, _ = self.policy_sample([obs])
        return actions[0]
    
    def update_parameters(self, memory, batch_size, updates):
        # Sample a batch from memory
        obs_batch, action_batch, reward_batch, next_obs_batch, mask_batch = memory.sample(batch_size=batch_size)
        # obs_batch
        # B x [rgb, rgbWoTargets, objectPatches]

        objects = action_batch[:, 0] - 1
        rotations = action_batch[:, 3] - 1

        B = obs_batch.shape[0]
        NB = len(obs_batch[0, 1])
        H, W = obs_batch[0, 0].shape[:2]
        rgbWoTargets = np.concatenate(obs_batch[:, 1]).reshape(B, NB, H, W, 3)
        state_q = rgbWoTargets[np.arange(B), objects]

        CH, CW = obs_batch[0, 2][0].shape[:2]
        objectPatches = np.concatenate(obs_batch[:, 2]).reshape(B, 2, NB, CH, CW, 3)
        patch = objectPatches[np.arange(B), rotations, objects]

        state_q = torch.FloatTensor(state_q).to(self.device)
        patch = torch.FloatTensor(patch).to(self.device)

        reward_batch = torch.FloatTensor(reward_batch).to(self.device) #.unsqueeze(1)
        mask_batch = torch.FloatTensor(mask_batch).to(self.device) #.unsqueeze(1)

        with torch.no_grad():
            next_action, next_state_log_pi = self.policy_sample(next_obs_batch)
            
            next_objects = next_action[:, 0] - 1
            next_rotations = next_action[:, 3] - 1

            next_rgbWoTargets = np.concatenate(next_obs_batch[:, 1]).reshape(B, NB, H, W, 3)
            next_state_q = next_rgbWoTargets[np.arange(B), next_objects]

            next_objectPatches = np.concatenate(next_obs_batch[:, 2]).reshape(B, 2, NB, CH, CW, 3)
            next_patch = next_objectPatches[np.arange(B), next_rotations, next_objects]

            next_state_q = torch.FloatTensor(next_state_q).to(self.device)
            next_patch = torch.FloatTensor(next_patch).to(self.device)

            qf1_next_values, qf2_next_values = self.critic_target.both([None, next_state_q, next_patch])
            qf1_next_target = qf1_next_values[torch.arange(B), next_action[:, 1], next_action[:, 2]]
            qf2_next_target = qf2_next_values[torch.arange(B), next_action[:, 1], next_action[:, 2]]

            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
            next_q_value = reward_batch + mask_batch * self.gamma * (min_qf_next_target)
            
        qf1_values, qf2_values = self.critic.both([None, state_q, patch])
        qf1 = qf1_values[torch.arange(B), action_batch[:, 1], action_batch[:, 2]]
        qf2 = qf2_values[torch.arange(B), action_batch[:, 1], action_batch[:, 2]]
        qf1_loss = F.mse_loss(qf1, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf2_loss = F.mse_loss(qf2, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf_loss = qf1_loss + qf2_loss
        
        self.critic_optim.zero_grad()
        qf_loss.backward()
        self.critic_optim.step()

        action_pi, log_pi = self.policy_sample(obs_batch)
        with torch.no_grad():
            qf1_pi_values, qf2_pi_values = self.critic.both([None, state_q, patch])
        qf1_pi = qf1_pi_values[torch.arange(B), action_pi[:, 1], action_pi[:, 2]]
        qf2_pi = qf2_pi_values[torch.arange(B), action_pi[:, 1], action_pi[:, 2]]
        min_qf_pi = torch.min(qf1_pi, qf2_pi)
        policy_loss = ((self.alpha * log_pi) - min_qf_pi).mean() # Jπ = 𝔼st∼D,εt∼N[α * logπ(f(εt;st)|st) − Q(st,f(εt;st))]

        self.policy_place_optim.zero_grad()
        policy_loss.backward()
        self.policy_place_optim.step()

        if self.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()

            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()

            self.alpha = self.log_alpha.exp()
            alpha_tlogs = self.alpha.clone() # For TensorboardX logs
        else:
            alpha_loss = torch.tensor(0.).to(self.device)
            alpha_tlogs = torch.tensor(self.alpha) # For TensorboardX logs


        if updates % self.target_update_interval == 0:
            soft_update(self.critic_target, self.critic, self.tau)

        return qf1_loss.item(), qf2_loss.item(), policy_loss.item(), alpha_loss.item(), alpha_tlogs.item()

    # Save model parameters
    def save_checkpoint(self, log_name, suffix="", ckpt_path=None):
        if not os.path.exists('checkpoints/'):
            os.makedirs('checkpoints/')
        if ckpt_path is None:
            ckpt_path = "checkpoints/sac_{}_{}.pth".format(log_name, suffix)
        print('Saving models to {}'.format(ckpt_path))
        torch.save({'policy_state_dict': self.policy_place.state_dict(),
                    'critic_state_dict': self.critic.state_dict(),
                    'critic_target_state_dict': self.critic_target.state_dict(),
                    'critic_optimizer_state_dict': self.critic_optim.state_dict(),
                    'policy_optimizer_state_dict': self.policy_place_optim.state_dict()}, ckpt_path)

    # Load model parameters
    def load_checkpoint(self, ckpt_path, evaluate=False):
        print('Loading models from {}'.format(ckpt_path))
        if ckpt_path is not None:
            checkpoint = torch.load(ckpt_path)
            self.policy_place.load_state_dict(checkpoint['policy_state_dict'])
            self.critic.load_state_dict(checkpoint['critic_state_dict'])
            self.critic_target.load_state_dict(checkpoint['critic_target_state_dict'])
            self.critic_optim.load_state_dict(checkpoint['critic_optimizer_state_dict'])
            self.policy_place_optim.load_state_dict(checkpoint['policy_optimizer_state_dict'])

            if evaluate:
                self.policy_place.eval()
                self.critic.eval()
                self.critic_target.eval()
            else:
                self.policy_place.train()
                self.critic.train()
                self.critic_target.train()

