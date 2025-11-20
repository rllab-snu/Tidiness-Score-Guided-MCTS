import torch
import torch.nn as nn
from .util import mlp, resnet
from torchvision.models import resnet18
from src.models.transport_small import TransportSmall

class ResNetP(nn.Module):
    def __init__(self, hidden_dim=64):
        super().__init__()
        resnet = resnet18(pretrained=False)
        self.cnn_state = nn.Sequential(
            nn.Conv2d(6, 64, kernel_size=7, stride=2, padding=3),
            *list(resnet.children())[1:-1]
            )
        self.fc = nn.Sequential(
                                nn.Linear(512, hidden_dim),
                                nn.ReLU(),
                                nn.Linear(hidden_dim, 1),
                                nn.Tanh(),
                        )
        
    def forward(self, state):
        state = state.permute(0, 3, 1, 2)
        h = self.cnn_state(state) # B x 6 x H x W
        h = h.view(-1, 512)
        q = self.fc(h) # B x 1

        q = q.view(-1)
        q = torch.softmax(q, dim=-1)
        return q

class TransportQ(nn.Module):
    def __init__(self, crop_size=64):
        super().__init__()
        self.q1 = TransportSmall(
            in_channels=3, 
            n_rotations=1, #8 
            crop_size=crop_size, 
            verbose=False,
            name="Transport-Q1")
        self.q2 = TransportSmall(
            in_channels=3, 
            n_rotations=1, #8 
            crop_size=crop_size, 
            verbose=False,
            name="Transport-Q2")

    def both(self, obs):
        _, state_q, patch = obs
        return self.q1(state_q, patch), self.q2(state_q, patch)

    def forward(self, obs):
        q1, q2 = self.both(obs)
        #print(q1.shape, q2.shape)
        return torch.min(q1, q2)

class ResNetQ(nn.Module):
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
        
    def forward(self, state, patch):
        state = state.permute(0, 3, 1, 2)
        patch = patch.permute(0, 3, 1, 2)
        h_state = self.cnn_state(state) # B x 32 x H x W
        f_patch = self.cnn_patch(patch) # B x 32
        f_patch = f_patch.unsqueeze(-1).unsqueeze(-1)
        f_patch = f_patch.expand(-1, -1, h_state.size(2), h_state.size(3))
        h_cat = torch.cat([h_state, f_patch], dim=1)  # B x 64 x H x W
        q = self.fconv(h_cat).squeeze(1)  # B x H x W
        return q

class ResNetTwinQ(nn.Module):
    def __init__(self, crop_size=64):
        super().__init__()
        
        self.q1 = ResNetQ(hidden_dim=32)
        self.q2 = ResNetQ(hidden_dim=32)

    def both(self, obs):
        _, state_q, patch = obs
        return self.q1(state_q, patch), self.q2(state_q, patch)

    def forward(self, obs):
        q1, q2 = self.both(obs)
        #print(q1.shape, q2.shape)
        return torch.min(q1, q2)

class ValueFunction(nn.Module):
    def __init__(self, hidden_dim=16):
        super().__init__()
        self.v = resnet(
            num_blocks=4,
            in_channels=3,
            out_channels=4,
            hidden_dim=hidden_dim,
            output_activation=None,
            stride=2
            )
        self.fc1 = nn.Linear(4 * 23 * 30, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, obs):
        state_v, _, _ = obs
        state_v = state_v.permute(0, 3, 1, 2)
        h = self.v(state_v)
        h = h.reshape(-1, 4 * 23 * 30)
        h = torch.relu(self.fc1(h))
        out = self.fc2(h)
        return out

class RewardFunction(nn.Module):
    def __init__(self, hidden_dim=16, sigmoid=False):
        super().__init__()
        self.v = resnet(
            num_blocks=4,
            in_channels=3,
            out_channels=4,
            hidden_dim=hidden_dim,
            output_activation=None,
            stride=2
            )
        self.fc1 = nn.Linear(4 * 23 * 30, 128)
        self.fc2 = nn.Linear(128, 1)
        self.sigmoid = sigmoid

    def forward(self, obs):
        state_v, _, _ = obs
        state_v = state_v.permute(0, 3, 1, 2)
        h = self.v(state_v)
        h = h.reshape(-1, 4 * 23 * 30)
        h = torch.relu(self.fc1(h))
        out = self.fc2(h)
        if self.sigmoid:
            out = torch.sigmoid(out)
        return out
