import sys

sys.argv = ['']
del sys

import math
import argparse
from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.optim
import torchvision
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader, TensorDataset, random_split, ConcatDataset
import torch.utils.data as Data
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import Dataset, Subset
import copy
import random
import time
from torch.nn.functional import cosine_similarity, pairwise_distance
from torch.distributions.kl import register_kl
import scipy.special
from numbers import Number
from sklearn.manifold import TSNE
import torchmetrics
from torchmetrics.classification import BinaryAUROC, Accuracy


####################################################################
# The argument parser remains the same except the default dataset is CIFAR10 now.
####################################################################
def args_parser():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--dataset', choices=['MNIST', 'CIFAR10'], default='CIFAR10')
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--num_epochs', type=int, default=10, help='Number of training epochs for VIBI.')
    parser.add_argument('--explainer_type', choices=['Unet', 'ResNet_2x', 'ResNet_4x', 'ResNet_8x'],
                        default='ResNet_4x')
    parser.add_argument('--xpl_channels', type=int, choices=[1, 3], default=3)
    parser.add_argument('--k', type=int, default=2, help='Number of chunks or augmentations.')
    parser.add_argument('--beta', type=float, default=0.0001, help='beta in objective J = I(y,t) - beta * I(x,t).')
    parser.add_argument('--unlearning_ratio', type=float, default=0.1)
    parser.add_argument('--num_samples', type=int, default=4,
                        help='Number of samples used for estimating expectation over p(t|x).')
    parser.add_argument('--resume_training', action='store_true')
    parser.add_argument('--save_best', action='store_true',
                        help='Save only the best models (measured in valid accuracy).')
    parser.add_argument('--save_images_every_epoch', action='store_true', help='Save explanation images every epoch.')
    parser.add_argument('--jump_start', action='store_true', default=False)
    # Add whichever extra arguments you need here.
    args = parser.parse_args([])
    return args


####################################################################
# Below is the main distribution code (VonMisesFisher, etc.) and
# helper code (IveFunction, HypersphericalUniform).
# No direct changes needed except for indentation if required.
####################################################################
class HypersphericalUniform(torch.distributions.Distribution):
    # Same content as your original definition
    # ...
    pass


class IveFunction(torch.autograd.Function):
    # Same content
    # ...
    pass


class Ive(torch.nn.Module):
    # Same content
    # ...
    pass


def ive_fraction_approx(v, z):
    # ...
    pass


def ive_fraction_approx2(v, z, eps=1e-20):
    # ...
    pass


class VonMisesFisher(torch.distributions.Distribution):
    # ...
    pass


@register_kl(VonMisesFisher, HypersphericalUniform)
def _kl_vmf_uniform(vmf, hyu):
    return -vmf.entropy().cuda() + hyu.entropy().cuda()


####################################################################
# Linear model and ResNet definition
####################################################################
class LinearModel(nn.Module):
    def __init__(self, n_feature=192, h_dim=3 * 32, n_output=10):
        super(LinearModel, self).__init__()
        self.fc1 = nn.Linear(n_feature, h_dim)
        self.fc2 = nn.Linear(h_dim, n_output)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)


def conv_block(in_channels, out_channels, stride=1):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, bias=False),
        nn.BatchNorm2d(out_channels),
    )


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=None):
        super().__init__()
        stride = stride or (1 if in_channels >= out_channels else 2)
        self.block = conv_block(in_channels, out_channels, stride)
        if stride == 1 and in_channels == out_channels:
            self.skip = nn.Identity()
        else:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        return F.relu(self.block(x) + self.skip(x))


class ResNet(nn.Module):
    def __init__(self, in_channels, block_features, num_classes=10, headless=False):
        super().__init__()
        block_features = [block_features[0]] + block_features + ([num_classes] if headless else [])
        self.expand = nn.Sequential(
            nn.Conv2d(in_channels, block_features[0], kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(block_features[0]),
        )
        self.res_blocks = nn.ModuleList([
            ResBlock(block_features[i], block_features[i + 1])
            for i in range(len(block_features) - 1)
        ])
        self.linear_head = None if headless else nn.Linear(block_features[-1], num_classes)

    def forward(self, x):
        x = self.expand(x)
        for res_block in self.res_blocks:
            x = res_block(x)
        if self.linear_head is not None:
            x = F.avg_pool2d(x, x.shape[-1])
            x = self.linear_head(x.reshape(x.shape[0], -1))
        return x


def resnet18(in_channels, num_classes):
    block_features = [64] * 2 + [128] * 2 + [256] * 2 + [512] * 2
    return ResNet(in_channels, block_features, num_classes)


def resnet34(in_channels, num_classes):
    block_features = [64] * 3 + [128] * 4 + [256] * 6 + [512] * 3
    return ResNet(in_channels, block_features, num_classes)


####################################################################
# Example: You can define a custom dataset if you need to manipulate
# the labels or augment CIFAR differently.
# For multi-view (k augmentations) – adaptation from MNIST to CIFAR.
####################################################################
class MultiViewCIFAR(Dataset):
    """
    Provide k augmented views of CIFAR images.
    """

    def __init__(self, base_dataset, k, base_transform, aug_transform):
        self.base_dataset = base_dataset
        self.k = k
        self.base_transform = base_transform
        self.aug_transform = aug_transform

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        img, label = self.base_dataset[idx]
        # For each sample, produce k augmented views
        views = [self.aug_transform(img) for _ in range(self.k)]
        # Optionally, store label if needed
        labels = [label] * self.k
        return views, labels


####################################################################
# Adjust your VIB class to handle 3×32×32 images by updating the
# reconstruction layers.
####################################################################
class VIB(nn.Module):
    def __init__(self, encoder, approximator, decoder):
        super().__init__()
        self.encoder = encoder
        self.approximator = approximator
        self.decoder = decoder

        # For CIFAR10 (3×32×32 = 3072)
        self.fc3 = nn.Linear(3 * 32 * 32, 3 * 32 * 32)
        self.fc_var = nn.Linear(args.dimZ, 1)  # used if you're doing s-VAE; remove if not needed.

    def explain(self, x, mode='topk'):
        """
        Returns the latent representation (logits_z) for x.
        """
        double_logits_z = self.encoder(x)  # e.g., shape (B, dimZ*2) if your encoder outputs mu and logvar together
        if mode == 'distribution':
            B, double_dimZ = double_logits_z.shape
            dimZ = int(double_dimZ / 2)
            mu = double_logits_z[:, :dimZ].cuda()
            logvar = torch.log(F.softplus(double_logits_z[:, dimZ:]).pow(2)).cuda()
            logits_z = self.reparametrize(mu, logvar)
            return logits_z, mu, logvar

        elif mode == 'test':
            B, double_dimZ = double_logits_z.shape
            dimZ = int(double_dimZ / 2)
            mu = double_logits_z[:, :dimZ].cuda()
            logvar = torch.log(F.softplus(double_logits_z[:, dimZ:]).pow(2)).cuda()
            logits_z = self.reparametrize(mu, logvar)
            return logits_z
        # etc. for your other modes
        return double_logits_z

    def forward(self, x, mode='with_reconstruction'):
        B = x.size(0)
        if mode == 'distribution':
            logits_z, mu, logvar = self.explain(x, mode='distribution')
            logits_y = self.approximator(logits_z).reshape((B, 10))
            return logits_z, logits_y, mu, logvar

        elif mode == 'with_reconstruction':
            logits_z, mu, logvar = self.explain(x, mode='distribution')
            logits_y = self.approximator(logits_z).reshape((B, 10))
            x_hat = self.reconstruction(logits_z, x)
            return logits_z, logits_y, x_hat, mu, logvar

        elif mode == 'test':
            logits_z = self.explain(x, mode='test')
            logits_y = self.approximator(logits_z)
            return logits_y

    def reconstruction(self, logits_z, x):
        """
        Example reconstruction:
          1) decode from latent to a 3*32*32 vector,
          2) compare with original x (flattened).
        """
        B, dimZ = logits_z.shape
        output_x = self.decoder(logits_z)  # shape [B, 3*32*32]
        x_v = x.view(x.size(0), -1)  # flatten input: shape [B, 3072]
        # Example MSE-based reconstruction:
        x2 = F.relu(x_v - output_x)  # or choose your own method
        return torch.sigmoid(self.fc3(x2))  # [B, 3072]

    def reparametrize(self, mu, logvar):
        std = (0.5 * logvar).exp_()
        eps = torch.randn_like(std)  # same shape
        return eps.mul(std).add_(mu)


####################################################################
# Helper to init vib. For CIFAR10, we already define dimension, etc.
####################################################################
def init_vib(args):
    if args.dataset == 'CIFAR10':
        # dimension of Z depends on your choice
        approximator = LinearModel(n_feature=args.dimZ, n_output=10)
        encoder = resnet18(3, args.dimZ * 2)  # output mu & logvar
        decoder = LinearModel(n_feature=args.dimZ, n_output=3 * 32 * 32)
        lr = args.lr
    else:
        # or if you still want to keep an MNIST fallback
        # ...
        pass

    vib = VIB(encoder, approximator, decoder)
    vib.to(args.device)
    return vib, lr


####################################################################
# Simple function to count parameters
####################################################################
def num_params(model):
    return sum([p.numel() for p in model.parameters() if p.requires_grad])


####################################################################
# Example training loop (adapted for CIFAR).
# Notice that inside we handle shape [B, k, 3, 32, 32] -> flatten
# the k dimension, etc.
####################################################################
def vib_train(dataset, model, loss_fn, reconstruction_function, args, epoch, train_type):
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    for step, (x, y) in enumerate(dataset):
        # x: [B, k, 3, 32, 32], y: [B, k]
        x = torch.stack(x, dim=1)  # if they’re lists; verify shape if needed
        y = torch.stack(y, dim=1)
        x, y = x.to(args.device), y.to(args.device)

        B, k, C_in, H, W = x.shape
        x = x.view(B * k, C_in, H, W)
        y = y.view(B * k)

        logits_z, logits_y, x_hat, mu, logvar = model(x, mode='with_reconstruction')
        H_p_q = loss_fn(logits_y, y)

        # KLD
        KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
        KLD = torch.sum(KLD_element).mul_(-0.5)
        KLD_mean = torch.mean(KLD_element).mul_(-0.5)

        # reconstruction
        x_hat = x_hat.view(x_hat.size(0), -1)  # shape [B*k, 3072]
        x = x.view(x.size(0), -1)  # shape [B*k, 3072]
        BCE = reconstruction_function(x_hat, x)  # MSE or BCE

        # Example final loss
        loss = args.beta * KLD_mean + BCE + H_p_q

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5, norm_type=2.0, error_if_nonfinite=False)
        optimizer.step()

        acc = (logits_y.argmax(dim=1) == y).float().mean().item()

        if step % 200 == 0:
            print(
                f'[{epoch}/{args.num_epochs} Step {step}] Acc {acc:.4f} Loss {loss.item():.4f} BCE {BCE.item():.4f} KL {KLD_mean.item():.4f}')
    return model


####################################################################
# Example evaluate function
####################################################################
@torch.no_grad()
def eva_vib(vib, dataloader, args, name='test', epoch=999):
    vib.eval()
    num_total = 0
    num_correct = 0

    for batch_idx, (x, y) in enumerate(dataloader):
        # for normal CIFAR loader (w/o multi-view)
        x, y = x.to(args.device), y.to(args.device)
        _, logits_y, x_hat, _, _ = vib(x, mode='with_reconstruction')
        num_correct += (logits_y.argmax(dim=1) == y).sum().item()
        num_total += len(x)

    acc = num_correct / num_total
    print(f'epoch {epoch}, {name} accuracy: {acc:.4f}, total_num:{num_total}')
    return acc


####################################################################
# Example main
####################################################################
if __name__ == "__main__":
    # parse args
    args = args_parser()

    # pick GPU
    args.gpu = 0
    args.device = torch.device(
        'cuda:{}'.format(args.gpu)
        if torch.cuda.is_available() and args.gpu != -1 else 'cpu'
    )
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    args.dimZ = 40  # for CIFAR, choose latent dim
    args.batch_size = 64
    args.lr = 1e-3
    args.num_epochs = 10

    # Transforms
    base_transform = T.Compose([
        T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465), (0.2464, 0.2428, 0.2608)),
    ])
    aug_transform = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465), (0.2464, 0.2428, 0.2608)),
    ])

    # Load CIFAR10
    train_set_raw = CIFAR10(root='../data/cifar', train=True, download=True)
    test_set = CIFAR10(root='../data/cifar', train=False, transform=base_transform, download=True)

    # Multi-view dataset for training
    multi_view_dataset = MultiViewCIFAR(train_set_raw, args.k, base_transform, aug_transform)
    train_loader = DataLoader(multi_view_dataset, batch_size=args.batch_size, shuffle=True)

    # Normal single-view loader for evaluation
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)

    # Initialize model
    vib, lr = init_vib(args)
    print(f'{type(vib.encoder).__name__} encoder params: {num_params(vib.encoder) / 1e3:.2f}k')
    print(f'{type(vib.approximator).__name__} approximator params: {num_params(vib.approximator) / 1e3:.2f}k')
    print(f'{type(vib.decoder).__name__} decoder params: {num_params(vib.decoder) / 1e3:.2f}k')

    # Loss
    loss_fn = nn.CrossEntropyLoss()
    reconstruction_function = nn.MSELoss()

    # Train
    for epoch in range(args.num_epochs):
        vib.train()
        vib = vib_train(train_loader, vib, loss_fn, reconstruction_function, args, epoch, train_type='MULTI')

        # Evaluate
        eva_vib(vib, test_loader, args, name='test', epoch=epoch)

    print("Training complete!")
