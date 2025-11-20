import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
from torch.distributions.categorical import Categorical
from torchvision.models import resnet18

from .models.transport_small import TransportSmall
from .value_functions import ResNetQ, ResNetP

from .util import mlp


LOG_STD_MIN = -5.0
LOG_STD_MAX = 2.0


class PickPolicy(nn.Module):
    def __init__(self, crop_size=64):
        super().__init__()
        self.p = ResNetP(hidden_dim=32)

    def forward(self, obs):
        state_v, state_q, _ = obs
        N = state_q.size(0)
        state_v = state_v.unsqueeze(0).repeat([N, 1, 1, 1])
        state = torch.cat([state_v, state_q], dim=-1)

        action_probs = self.p(state)
        dist = Categorical(action_probs)
        action = dist.sample()

        action_probs = action_probs.clamp(min=1e-8, max=1.0)
        log_action_probs = torch.log(action_probs) # + z)
        return action, action_probs, log_action_probs, dist.log_prob(action)

    def get_prob(self, obs):
        state_v, state_q, _ = obs
        N = state_q.size(0)
        state_v = state_v.unsqueeze(0).repeat([N, 1, 1, 1])
        state = torch.cat([state_v, state_q], dim=-1)

        action_probs = self.p(state)
        #action_probs = action_probs.clamp(min=1e-8, max=1.0)
        return action_probs
    
    
class DiscreteTransportPolicy(nn.Module):
    def __init__(self, crop_size=64):
        super().__init__()
        self.q = TransportSmall(
            in_channels=3, 
            n_rotations=1, #8 
            crop_size=crop_size, 
            verbose=False,
            name="Policy-Q")

    def forward(self, obs):
        _, state_q, patch = obs
        action_probs = self.q(state_q, patch, softmax=True)
        B, H, W = action_probs.size()
        C = 1
        action_probs_flatten = action_probs.view(B, -1)
        dist = Categorical(action_probs_flatten)
        actions_flatten = dist.sample()
        a0 = actions_flatten // (W * C)
        a1 = (actions_flatten % (W * C)) // C
        a2 = actions_flatten % C
        actions = torch.stack([a0, a1, a2], dim=-1)
        actions = actions.to(state_q.device)
        # actions = dist.sample().to(state.device)

        # Have to deal with situation of 0.0 probabilities because we can't do log 0
        # z = action_probs == 0.0
        # z = z.float() * 1e-8
        action_probs = action_probs.clamp(min=1e-8, max=1.0)
        log_action_probs = torch.log(action_probs) # + z)

        return actions, action_probs, log_action_probs, dist.log_prob(actions_flatten)
    

class DeterministicTransportPolicy(nn.Module):
    def __init__(self, crop_size=64):
        super().__init__()
        self.q = TransportSmall(
            in_channels=3, 
            n_rotations=1, #8 
            crop_size=crop_size, 
            verbose=False,
            name="Policy-Q")

    def forward(self, obs):
        _, state_q, patch = obs
        action_probs = self.q(state_q, patch, softmax=True)
        B, H, W = action_probs.size()
        C = 1
        action_probs_flatten = action_probs.view(B, -1)
        actions_flatten = torch.argmax(action_probs_flatten, dim=-1)
        a0 = actions_flatten // (W * C)
        a1 = (actions_flatten % (W * C)) // C
        a2 = actions_flatten % C
        actions = torch.stack([a0, a1, a2], dim=-1)
        actions = actions.to(state_q.device)

        # Have to deal with situation of 0.0 probabilities because we can't do log 0
        z = action_probs == 0.0
        z = z.float() * 1e-8
        log_action_probs = torch.log(action_probs + z)
        return actions, action_probs, log_action_probs, dist.log_prob(actions_flatten)
    

class DiscreteResNetPolicy(nn.Module):
    def __init__(self, crop_size=64):
        super().__init__()
        self.q = ResNetQ(hidden_dim=32)

    def forward(self, obs):
        _, state_q, patch = obs
        q = self.q(state_q, patch)
        B, H, W = q.size()
        q_flat = q.view(B, H*W)
        action_probs_flatten = torch.softmax(q_flat, dim=-1)
        action_probs = action_probs_flatten.view(B, H, W)

        dist = Categorical(action_probs_flatten)
        actions_flatten = dist.sample()
        a0 = actions_flatten // W
        a1 = actions_flatten % W
        actions = torch.stack([a0, a1], dim=-1)
        actions = actions.to(state_q.device)
        # actions = dist.sample().to(state.device)

        # Have to deal with situation of 0.0 probabilities because we can't do log 0
        # z = action_probs == 0.0
        # z = z.float() * 1e-8
        action_probs = action_probs.clamp(min=1e-8, max=1.0)
        log_action_probs = torch.log(action_probs) # + z)

        return actions, action_probs, log_action_probs, dist.log_prob(actions_flatten)

    def get_prob(self, obs):
        _, state_q, patch = obs
        q = self.q(state_q, patch)
        B, H, W = q.size()
        q_flat = q.view(B, H*W)
        action_probs_flatten = torch.softmax(q_flat, dim=-1)
        action_probs = action_probs_flatten.view(B, H, W)
        #action_probs = action_probs.clamp(min=1e-8, max=1.0)
        return action_probs
    
class DeterministicResNetPolicy(nn.Module):
    def __init__(self, crop_size=64):
        super().__init__()
        self.q = ResNetQ(hidden_dim=32)

    def forward(self, obs):
        _, state_q, patch = obs
        q = self.q(state_q, patch)
        B, H, W = q.size()
        q_flat = q.view(B, H*W)
        action_probs_flatten = torch.softmax(q_flat, dim=-1)
        action_probs = action_probs_flatten.view(B, H, W)
        
        actions_flatten = torch.argmax(action_probs_flatten, dim=-1)
        a0 = actions_flatten // W
        a1 = actions_flatten % W
        actions = torch.stack([a0, a1], dim=-1)
        actions = actions.to(state_q.device)

        # Have to deal with situation of 0.0 probabilities because we can't do log 0
        z = action_probs == 0.0
        z = z.float() * 1e-8
        log_action_probs = torch.log(action_probs + z)
        return actions, action_probs, log_action_probs#, dist.log_prob(actions_flatten)
    
    def get_prob(self, obs):
        _, state_q, patch = obs
        q = self.q(state_q, patch)
        B, H, W = q.size()
        q_flat = q.view(B, H*W)
        action_probs_flatten = torch.softmax(q_flat, dim=-1)
        action_probs = action_probs_flatten.view(B, H, W)
        #action_probs = action_probs.clamp(min=1e-8, max=1.0)
        return action_probs
    
class GaussianPolicy(nn.Module):
    def __init__(self, crop_size=64):
        super().__init__()
        self.q = ResNetQ(hidden_dim=32)
        self.fc = nn.Linear(12*15, 2)
        self.log_std = nn.Parameter(torch.zeros(2, dtype=torch.float32))

    def forward(self, obs):
        _, state_q, patch = obs
        q = self.q(state_q, patch)
        B, H, W = q.size()
        q_flat = q.view(B, H*W)
        mean = self.fc(q_flat)
        std = torch.exp(self.log_std.clamp(LOG_STD_MIN, LOG_STD_MAX))
        scale_tril = torch.diag(std)
        return MultivariateNormal(mean, scale_tril=scale_tril)


# Option 0: same as original version
class PolicyOpt0(nn.Module):
    def __init__(self, hidden_dim=32):
        super().__init__()
        resnet = resnet18(pretrained=False)
        
        self.cnn_state = nn.Sequential(
            *list(resnet.children())[:-2]
            +[nn.Conv2d(512, hidden_dim, kernel_size=1, stride=1, padding=0)]
            )
        self.cnn_patch = resnet18(pretrained=False)
        self.cnn_patch.fc = nn.Sequential(
                                nn.Linear(512, hidden_dim),
                                nn.Tanh()
                            )
        # fully convolution layer
        self.fconv = nn.Sequential(
                        nn.Conv2d(2*hidden_dim, 4*hidden_dim, kernel_size=1, stride=1, padding='same'),
                        nn.ReLU(),
                        nn.Conv2d(4*hidden_dim, 1, kernel_size=1, stride=1, padding='same')
                        )

    def forward(self, obs):
        _, state, patch = obs
        state = state.permute(0, 3, 1, 2)
        patch = patch.permute(0, 3, 1, 2)
        h_state = self.cnn_state(state) # B x 32 x H x W
        f_patch = self.cnn_patch(patch) # B x 32
        f_patch = f_patch.unsqueeze(-1).unsqueeze(-1)
        f_patch = f_patch.expand(-1, -1, h_state.size(2), h_state.size(3))
        h_cat = torch.cat([h_state, f_patch], dim=1)  # B x 64 x H x W
        q = self.fconv(h_cat).squeeze(1)  # B x H x W

        B, H, W = q.size()
        q_flat = q.view(B, H*W)
        action_probs_flatten = torch.softmax(q_flat, dim=-1)
        action_probs = action_probs_flatten.view(B, H, W)

        dist = Categorical(action_probs_flatten)
        actions_flatten = dist.sample()
        a0 = actions_flatten // W
        a1 = actions_flatten % W
        actions = torch.stack([a0, a1], dim=-1)
        actions = actions.to(state.device)
        action_probs = action_probs.clamp(min=1e-8, max=1.0)
        log_action_probs = torch.log(action_probs) # + z)
        return actions, action_probs, log_action_probs, dist.log_prob(actions_flatten)
    
    def get_prob(self, obs):
        _, state, patch = obs
        state = state.permute(0, 3, 1, 2)
        patch = patch.permute(0, 3, 1, 2)
        h_state = self.cnn_state(state) # B x 32 x H x W
        f_patch = self.cnn_patch(patch) # B x 32
        f_patch = f_patch.unsqueeze(-1).unsqueeze(-1)
        f_patch = f_patch.expand(-1, -1, h_state.size(2), h_state.size(3))
        h_cat = torch.cat([h_state, f_patch], dim=1)  # B x 64 x H x W
        q = self.fconv(h_cat).squeeze(1)  # B x H x W

        B, H, W = q.size()
        q_flat = q.view(B, H*W)
        action_probs_flatten = torch.softmax(q_flat, dim=-1)
        action_probs = action_probs_flatten.view(B, H, W)
        return action_probs

# Option 1: without info of object patches
class PolicyOpt1(nn.Module):
    def __init__(self, hidden_dim=32):
        super().__init__()
        resnet = resnet18(pretrained=False)
        
        self.cnn_state = nn.Sequential(
            *list(resnet.children())[:-2]
            +[nn.Conv2d(512, hidden_dim, kernel_size=1, stride=1, padding=0)]
            )
        
        # fully convolution layer
        self.fconv = nn.Sequential(
                        nn.Conv2d(hidden_dim, 4*hidden_dim, kernel_size=1, stride=1, padding='same'),
                        nn.ReLU(),
                        nn.Conv2d(4*hidden_dim, 1, kernel_size=1, stride=1, padding='same')
                        )
        
    def forward(self, obs):
        _, state, patch = obs
        state = state.permute(0, 3, 1, 2)
        h_state = self.cnn_state(state) # B x 32 x H x W
        q = self.fconv(h_state).squeeze(1)  # B x H x W

        B, H, W = q.size()
        q_flat = q.view(B, H*W)
        action_probs_flatten = torch.softmax(q_flat, dim=-1)
        action_probs = action_probs_flatten.view(B, H, W)

        dist = Categorical(action_probs_flatten)
        actions_flatten = dist.sample()
        a0 = actions_flatten // W
        a1 = actions_flatten % W
        actions = torch.stack([a0, a1], dim=-1)
        actions = actions.to(state.device)
        action_probs = action_probs.clamp(min=1e-8, max=1.0)
        log_action_probs = torch.log(action_probs) # + z)
        return actions, action_probs, log_action_probs, dist.log_prob(actions_flatten)
    
    def get_prob(self, obs):
        _, state, patch = obs
        state = state.permute(0, 3, 1, 2)
        h_state = self.cnn_state(state) # B x 32 x H x W
        q = self.fconv(h_state).squeeze(1)  # B x H x W

        B, H, W = q.size()
        q_flat = q.view(B, H*W)
        action_probs_flatten = torch.softmax(q_flat, dim=-1)
        action_probs = action_probs_flatten.view(B, H, W)
        return action_probs

    def get_logits(self, obs):
        _, state, patch = obs
        state = state.permute(0, 3, 1, 2)
        h_state = self.cnn_state(state) # B x 32 x H x W
        q = self.fconv(h_state).squeeze(1)  # B x H x W

        B, H, W = q.size()
        logit = q.view(B, H*W)
        action_probs_flatten = torch.softmax(logit, dim=-1)
        action_probs = action_probs_flatten.view(B, H, W)
        return logit, action_probs

# Option 2: H,W of object patches
class PolicyOpt2(nn.Module):
    def __init__(self, hidden_dim=32):
        super().__init__()
        resnet = resnet18(pretrained=False)
        
        self.cnn_state = nn.Sequential(
            *list(resnet.children())[:-2]
            +[nn.Conv2d(512, hidden_dim, kernel_size=1, stride=1, padding=0)]
            )
        # fully convolution layer
        self.fconv = nn.Sequential(
                        nn.Conv2d(hidden_dim + 2, 4*hidden_dim, kernel_size=1, stride=1, padding='same'),
                        nn.ReLU(),
                        nn.Conv2d(4*hidden_dim, 1, kernel_size=1, stride=1, padding='same')
                        )
        
    def get_bbox_size(self, patches):
        bbox_size = torch.zeros(patches.size(0), 2).to(patches.device)
        for i, patch in enumerate(patches):
            py, px = torch.where(torch.sum(patch, dim=0)!=0)
            try:
                h = py.max() - py.min()
                w = px.max() - px.min()
            except:
                h, w = 0, 0
            bbox_size[i] = torch.tensor([h, w])/128
        return bbox_size

    def forward(self, obs):
        _, state, patch = obs
        state = state.permute(0, 3, 1, 2)
        patch = patch.permute(0, 3, 1, 2)
        h_state = self.cnn_state(state) # B x 32 x H x W
        f_patch = self.get_bbox_size(patch) # B x 2
        f_patch = f_patch.unsqueeze(-1).unsqueeze(-1)
        f_patch = f_patch.expand(-1, -1, h_state.size(2), h_state.size(3))
        h_cat = torch.cat([h_state, f_patch], dim=1)  # B x 34 x H x W
        q = self.fconv(h_cat).squeeze(1)  # B x H x W

        B, H, W = q.size()
        q_flat = q.view(B, H*W)
        action_probs_flatten = torch.softmax(q_flat, dim=-1)
        action_probs = action_probs_flatten.view(B, H, W)

        dist = Categorical(action_probs_flatten)
        actions_flatten = dist.sample()
        a0 = actions_flatten // W
        a1 = actions_flatten % W
        actions = torch.stack([a0, a1], dim=-1)
        actions = actions.to(state.device)
        action_probs = action_probs.clamp(min=1e-8, max=1.0)
        log_action_probs = torch.log(action_probs) # + z)
        return actions, action_probs, log_action_probs, dist.log_prob(actions_flatten)
    
    def get_prob(self, obs):
        _, state, patch = obs
        state = state.permute(0, 3, 1, 2)
        patch = patch.permute(0, 3, 1, 2)
        h_state = self.cnn_state(state) # B x 32 x H x W
        f_patch = self.get_bbox_size(patch) # B x 2
        f_patch = f_patch.unsqueeze(-1).unsqueeze(-1)
        f_patch = f_patch.expand(-1, -1, h_state.size(2), h_state.size(3))
        h_cat = torch.cat([h_state, f_patch], dim=1)  # B x 34 x H x W
        q = self.fconv(h_cat).squeeze(1)  # B x H x W

        B, H, W = q.size()
        q_flat = q.view(B, H*W)
        action_probs_flatten = torch.softmax(q_flat, dim=-1)
        action_probs = action_probs_flatten.view(B, H, W)
        return action_probs

# Option 3: Coordinate Convolution
class PolicyOpt3(nn.Module):
    def __init__(self, hidden_dim=32):
        super().__init__()
        resnet = resnet18(pretrained=False)
        
        self.cnn_state = nn.Sequential(
            *[CoordConv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)]
            +list(resnet.children())[1:-2]
            +[nn.Conv2d(512, hidden_dim, kernel_size=1, stride=1, padding=0)]
            )
        self.cnn_patch = resnet18(pretrained=False)
        self.cnn_patch.fc = nn.Sequential(
                                nn.Linear(512, hidden_dim),
                                nn.Tanh()
                            )
        # fully convolution layer
        self.fconv = nn.Sequential(
                        nn.Conv2d(2*hidden_dim, 4*hidden_dim, kernel_size=1, stride=1, padding='same'),
                        nn.ReLU(),
                        nn.Conv2d(4*hidden_dim, 1, kernel_size=1, stride=1, padding='same')
                        )

    def forward(self, obs):
        _, state, patch = obs
        state = state.permute(0, 3, 1, 2)
        patch = patch.permute(0, 3, 1, 2)
        h_state = self.cnn_state(state) # B x 32 x H x W
        f_patch = self.cnn_patch(patch) # B x 32
        f_patch = f_patch.unsqueeze(-1).unsqueeze(-1)
        f_patch = f_patch.expand(-1, -1, h_state.size(2), h_state.size(3))
        h_cat = torch.cat([h_state, f_patch], dim=1)  # B x 64 x H x W
        q = self.fconv(h_cat).squeeze(1)  # B x H x W

        B, H, W = q.size()
        q_flat = q.view(B, H*W)
        action_probs_flatten = torch.softmax(q_flat, dim=-1)
        action_probs = action_probs_flatten.view(B, H, W)

        dist = Categorical(action_probs_flatten)
        actions_flatten = dist.sample()
        a0 = actions_flatten // W
        a1 = actions_flatten % W
        actions = torch.stack([a0, a1], dim=-1)
        actions = actions.to(state.device)
        action_probs = action_probs.clamp(min=1e-8, max=1.0)
        log_action_probs = torch.log(action_probs) # + z)
        return actions, action_probs, log_action_probs, dist.log_prob(actions_flatten)
    
    def get_prob(self, obs):
        _, state, patch = obs
        state = state.permute(0, 3, 1, 2)
        patch = patch.permute(0, 3, 1, 2)
        h_state = self.cnn_state(state) # B x 32 x H x W
        f_patch = self.cnn_patch(patch) # B x 32
        f_patch = f_patch.unsqueeze(-1).unsqueeze(-1)
        f_patch = f_patch.expand(-1, -1, h_state.size(2), h_state.size(3))
        h_cat = torch.cat([h_state, f_patch], dim=1)  # B x 64 x H x W
        q = self.fconv(h_cat).squeeze(1)  # B x H x W

        B, H, W = q.size()
        q_flat = q.view(B, H*W)
        action_probs_flatten = torch.softmax(q_flat, dim=-1)
        action_probs = action_probs_flatten.view(B, H, W)
        return action_probs

# Option 4: H,W of object patches + Coordinate Convolution
class PolicyOpt4(nn.Module):
    def __init__(self, hidden_dim=32):
        super().__init__()
        resnet = resnet18(pretrained=False)
        
        self.cnn_state = nn.Sequential(
            *[CoordConv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)]
            +list(resnet.children())[1:-2]
            +[nn.Conv2d(512, hidden_dim, kernel_size=1, stride=1, padding=0)]
            )
        # fully convolution layer
        self.fconv = nn.Sequential(
                        nn.Conv2d(hidden_dim + 2, 4*hidden_dim, kernel_size=1, stride=1, padding='same'),
                        nn.ReLU(),
                        nn.Conv2d(4*hidden_dim, 1, kernel_size=1, stride=1, padding='same')
                        )
        
    def get_bbox_size(self, patches):
        bbox_size = torch.zeros(patches.size(0), 2).to(patches.device)
        for i, patch in enumerate(patches):
            py, px = torch.where(torch.sum(patch, dim=0)!=0)
            try:
                h = py.max() - py.min()
                w = px.max() - px.min()
            except:
                h, w = 0, 0
            bbox_size[i] = torch.tensor([h, w])/128
        return bbox_size

    def forward(self, obs):
        _, state, patch = obs
        state = state.permute(0, 3, 1, 2)
        patch = patch.permute(0, 3, 1, 2)
        h_state = self.cnn_state(state) # B x 32 x H x W
        f_patch = self.get_bbox_size(patch) # B x 2
        f_patch = f_patch.unsqueeze(-1).unsqueeze(-1)
        f_patch = f_patch.expand(-1, -1, h_state.size(2), h_state.size(3))
        h_cat = torch.cat([h_state, f_patch], dim=1)  # B x 34 x H x W
        q = self.fconv(h_cat).squeeze(1)  # B x H x W

        B, H, W = q.size()
        q_flat = q.view(B, H*W)
        action_probs_flatten = torch.softmax(q_flat, dim=-1)
        action_probs = action_probs_flatten.view(B, H, W)

        dist = Categorical(action_probs_flatten)
        actions_flatten = dist.sample()
        a0 = actions_flatten // W
        a1 = actions_flatten % W
        actions = torch.stack([a0, a1], dim=-1)
        actions = actions.to(state.device)
        action_probs = action_probs.clamp(min=1e-8, max=1.0)
        log_action_probs = torch.log(action_probs) # + z)
        return actions, action_probs, log_action_probs, dist.log_prob(actions_flatten)
    
    def get_prob(self, obs):
        _, state, patch = obs
        state = state.permute(0, 3, 1, 2)
        patch = patch.permute(0, 3, 1, 2)
        h_state = self.cnn_state(state) # B x 32 x H x W
        f_patch = self.get_bbox_size(patch) # B x 2
        f_patch = f_patch.unsqueeze(-1).unsqueeze(-1)
        f_patch = f_patch.expand(-1, -1, h_state.size(2), h_state.size(3))
        h_cat = torch.cat([h_state, f_patch], dim=1)  # B x 34 x H x W
        q = self.fconv(h_cat).squeeze(1)  # B x H x W

        B, H, W = q.size()
        q_flat = q.view(B, H*W)
        action_probs_flatten = torch.softmax(q_flat, dim=-1)
        action_probs = action_probs_flatten.view(B, H, W)
        return action_probs
    
# class GaussianPolicy(nn.Module):
#     def __init__(self, obs_dim, act_dim, hidden_dim=256, n_hidden=2):
#         super().__init__()
#         self.net = mlp([obs_dim, *([hidden_dim] * n_hidden), act_dim])
#         self.log_std = nn.Parameter(torch.zeros(act_dim, dtype=torch.float32))

#     def forward(self, obs):
#         mean = self.net(obs)
#         std = torch.exp(self.log_std.clamp(LOG_STD_MIN, LOG_STD_MAX))
#         scale_tril = torch.diag(std)
#         return MultivariateNormal(mean, scale_tril=scale_tril)
#         # if mean.ndim > 1:
#         #     batch_size = len(obs)
#         #     return MultivariateNormal(mean, scale_tril=scale_tril.repeat(batch_size, 1, 1))
#         # else:
#         #     return MultivariateNormal(mean, scale_tril=scale_tril)

#     def act(self, obs, deterministic=False, enable_grad=False):
#         with torch.set_grad_enabled(enable_grad):
#             dist = self(obs)
#             return dist.mean if deterministic else dist.sample()


# class DeterministicPolicy(nn.Module):
#     def __init__(self, obs_dim, act_dim, hidden_dim=256, n_hidden=2):
#         super().__init__()
#         self.net = mlp([obs_dim, *([hidden_dim] * n_hidden), act_dim],
#                        output_activation=nn.Tanh)

#     def forward(self, obs):
#         return self.net(obs)

#     def act(self, obs, deterministic=False, enable_grad=False):
#         with torch.set_grad_enabled(enable_grad):
#             return self(obs)

class CoordConv2d(nn.Conv2d):
    """
    2D Coordinate Convolution

    Source: An Intriguing Failing of Convolutional Neural Networks and the CoordConv Solution
    https://arxiv.org/abs/1807.03247
    (e.g. adds 2 channels per input feature map corresponding to (x, y) location on map)
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        padding_mode="zeros",
        coord_encoding="position",
    ):
        """
        Args:
            in_channels: number of channels of the input tensor [C, H, W]
            out_channels: number of output channels of the layer
            kernel_size: convolution kernel size
            stride: conv stride
            padding: conv padding
            dilation: conv dilation
            groups: conv groups
            bias: conv bias
            padding_mode: conv padding mode
            coord_encoding: type of coordinate encoding. currently only 'position' is implemented
        """

        assert coord_encoding in ["position"]
        self.coord_encoding = coord_encoding
        if coord_encoding == "position":
            in_channels += 2  # two extra channel for positional encoding
            self._position_enc = None  # position encoding
        else:
            raise Exception(
                "CoordConv2d: coord encoding {} not implemented".format(
                    self.coord_encoding
                )
            )
        nn.Conv2d.__init__(
            self,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
        )

    def output_shape(self, input_shape):
        """
        Function to compute output shape from inputs to this module.

        Args:
            input_shape (iterable of int): shape of input. Does not include batch dimension.
                Some modules may not need this argument, if their output does not depend
                on the size of the input, or if they assume fixed size input.

        Returns:
            out_shape ([int]): list of integers corresponding to output shape
        """

        # adds 2 to channel dimension
        return [input_shape[0] + 2] + input_shape[1:]

    def forward(self, input):
        b, c, h, w = input.shape
        if self.coord_encoding == "position":
            if self._position_enc is None:
                pos_y, pos_x = torch.meshgrid(torch.arange(h), torch.arange(w))
                pos_y = pos_y.float().to(input.device) / float(h)
                pos_x = pos_x.float().to(input.device) / float(w)
                self._position_enc = torch.stack((pos_y, pos_x)).unsqueeze(0)
            pos_enc = self._position_enc.expand(b, -1, -1, -1)
            input = torch.cat((input, pos_enc), dim=1)
        return super(CoordConv2d, self).forward(input)
