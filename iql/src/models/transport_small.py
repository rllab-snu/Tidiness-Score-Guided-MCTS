# coding=utf-8
# Adapted from Ravens - Transporter Networks, Zeng et al., 2021
# https://github.com/google-research/ravens

"""Transport module."""


import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange

#from src.models.resnet import ResNet43_8s, ResNet_small
from src.utils import utils, MeanMetrics, to_device
from src.utils.text import bold
from src.utils.utils import apply_rotations_to_tensor
#from src.util import resnet #_strides
from torchvision.models import resnet18


class TransportSmall(nn.Module):
    """Transport module."""
    def __init__(self, in_channels, n_rotations, crop_size, verbose=False, name="Transport"):
        super().__init__()
        """Transport module for placing.

        Args:
          in_shape: shape of input image.
          n_rotations: number of rotations of convolving kernel.
          crop_size: crop size around pick argmax used as convolving kernel.
        """
        self.iters = 0
        self.n_rotations = n_rotations
        self.crop_size = crop_size  # crop size must be N*16 (e.g. 96)

        self.pad_size = int(self.crop_size / 2)
        self.padding = np.zeros((3, 2), dtype=int)
        self.padding[:2, :] = self.pad_size

        # Crop before network (default for Transporters in CoRL submission).
        if not hasattr(self, 'output_dim'):
            self.output_dim = 3
        if not hasattr(self, 'kernel_dim'):
            self.kernel_dim = 3

        use_ResNet = True
        if use_ResNet:
            resnet = resnet18(pretrained=True)
            self.model_query = nn.Sequential(
                *list(resnet.children())[:-2]
                +[nn.Conv2d(512, 32, kernel_size=1, stride=1, padding=0)]
                )
            resnet = resnet18(pretrained=True)
            self.model_key = nn.Sequential(
                *list(resnet.children())[:-2]
                +[nn.Conv2d(512, 32, kernel_size=1, stride=1, padding=0)]
                )
        else:
            hidden_dim = 16
            self.model_query = nn.Sequential(
                nn.Conv2d(in_channels, hidden_dim, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU(),
                nn.Conv2d(hidden_dim, 2*hidden_dim, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(2*hidden_dim),
                nn.ReLU(),
                nn.Conv2d(2*hidden_dim, 4*hidden_dim, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(4*hidden_dim),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=3, stride=3, padding=1),
                nn.Conv2d(4*hidden_dim, 4*hidden_dim, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(4*hidden_dim),
                nn.ReLU(),
                nn.Conv2d(4*hidden_dim, 4 * hidden_dim, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(4 * hidden_dim),
                nn.ReLU(),
                nn.Conv2d(4 * hidden_dim, 4 * hidden_dim, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(4 * hidden_dim),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=3, stride=3, padding=1),
            )
            self.model_key = nn.Sequential(
                nn.Conv2d(in_channels, hidden_dim, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU(),
                nn.Conv2d(hidden_dim, 2 * hidden_dim, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(2 * hidden_dim),
                nn.ReLU(),
                nn.Conv2d(2 * hidden_dim, 4 * hidden_dim, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(4 * hidden_dim),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=3, stride=3, padding=1),
                nn.Conv2d(4 * hidden_dim, 4 * hidden_dim, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(4 * hidden_dim),
                nn.ReLU(),
                nn.Conv2d(4*hidden_dim, 4 * hidden_dim, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(4 * hidden_dim),
                nn.ReLU(),
                nn.Conv2d(4 * hidden_dim, 4 * hidden_dim, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(4 * hidden_dim),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=3, stride=3, padding=1),
            )
        # self.model_query = resnet(num_blocks=4, in_channels=3, out_channels=32, hidden_dim=16,
        #                                   output_activation=None)#, strides=[2, 2, 3, 3])
        # self.model_key = resnet(num_blocks=4, in_channels=3, out_channels=32, hidden_dim=16,
        #                                 output_activation=None)#, strides=[2, 2, 3, 3])

        self.device = to_device(
            [self.model_query, self.model_key], name, verbose=verbose)

        self.optimizer_query = optim.Adam(
            self.model_query.parameters(), lr=1e-4)
        self.optimizer_key = optim.Adam(self.model_key.parameters(), lr=1e-4)
        self.loss = torch.nn.CrossEntropyLoss(reduction='mean')

        self.metric = MeanMetrics()

        self.softmax = nn.Softmax(dim=1)

    def correlate(self, in0, in1, softmax):
        """Correlate two input tensors."""
        output = F.conv2d(in0, in1, padding='same')

        if softmax:
            output_shape = output.shape
            output = Rearrange('b c h w -> b (c h w)')(output)
            #output = output.clamp(min=1e-4)
            output = self.softmax(output)
            output = Rearrange(
                'b (c h w) -> b h w c',
                c=output_shape[1],
                h=output_shape[2],
                w=output_shape[3])(output)
        else:
            output = Rearrange('b c h w -> b h w c')(output)
        return output[:, :, :, 0]

    def forward(self, in_img, patch, softmax=False):
        in_tensor = (in_img).to(torch.float32).to(self.device)
        crop = (patch).to(torch.float32).to(self.device)
        in_tensor = Rearrange('b h w c -> b c h w')(in_tensor)
        crop = Rearrange('b h w c -> b c h w')(crop)

        logits = self.model_query(in_tensor)
        kernel = self.model_key(crop)
        # logits = F.interpolate(logits, scale_factor=1/4, mode='bilinear')
        # kernel = F.interpolate(kernel, scale_factor=1/4, mode='bilinear')

        return self.correlate(logits, kernel, softmax)

    def train_block(self, in_img, patch, q, theta):
        output = self.forward(in_img, patch, softmax=False)

        itheta = theta / (2 * np.pi / self.n_rotations)
        itheta = np.int32(np.round(itheta)) % self.n_rotations

        # Get one-hot pixel label map.
        label_size = in_img.shape[:2] + (self.n_rotations,)
        label = np.zeros(label_size)
        label[q[0], q[1], itheta] = 1

        # Get loss.
        label = Rearrange('h w c -> 1 (h w c)')(label)
        label = torch.tensor(label, dtype=torch.float32).to(self.device)
        label = torch.argmax(label, dim=1)
        output = Rearrange('b theta h w -> b (h w theta)')(output)

        loss = self.loss(output, label)

        return loss

    def train(self, in_img, patch, q, theta):
        """Transport patch to pixel q.

        Args:
          in_img: input image.
          patch: patch image
          q: pixel (y, x)
          theta: rotation label in radians.
          backprop: True if backpropagating gradients.

        Returns:
          loss: training loss.
        """

        self.metric.reset()
        self.train_mode()
        self.optimizer_query.zero_grad()
        self.optimizer_key.zero_grad()

        loss = self.train_block(in_img, patch, q, theta)
        loss.backward()
        self.optimizer_query.step()
        self.optimizer_key.step()
        self.metric(loss)

        self.iters += 1
        return np.float32(loss.detach().cpu().numpy())

    def test(self, in_img, patch, q, theta):
        """Test."""
        self.eval_mode()

        with torch.no_grad():
            loss = self.train_block(in_img, patch, q, theta)

        self.iters += 1
        return np.float32(loss.detach().cpu().numpy())

    def train_mode(self):
        self.model_query.train()
        self.model_key.train()

    def eval_mode(self):
        self.model_query.eval()
        self.model_key.eval()

    def format_fname(self, fname, is_query):
        suffix = 'query' if is_query else 'key'
        return fname.split('.pth')[0] + f'_{suffix}.pth'

    def load(self, fname, verbose):
        query_name = self.format_fname(fname, is_query=True)
        key_name = self.format_fname(fname, is_query=False)

        if verbose:
            device = "GPU" if self.device.type == "cuda" else "CPU"
            print(
                f"Loading {bold('transport query')} model on {bold(device)} from {bold(query_name)}")
            print(
                f"Loading {bold('transport key')}   model on {bold(device)} from {bold(key_name)}")

        self.model_query.load_state_dict(
            torch.load(query_name, map_location=self.device))
        self.model_key.load_state_dict(
            torch.load(key_name, map_location=self.device))

    def save(self, fname, verbose=False):
        query_name = self.format_fname(fname, is_query=True)
        key_name = self.format_fname(fname, is_query=False)

        if verbose:
            print(
                f"Saving {bold('transport query')} model to {bold(query_name)}")
            print(
                f"Saving {bold('transport key')}   model to {bold(key_name)}")

        torch.save(self.model_query.state_dict(), query_name)
        torch.save(self.model_key.state_dict(), key_name)
