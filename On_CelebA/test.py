import sys
sys.argv = ['']
del sys

import math
import argparse
import torch
import torch.nn as nn
import torch.optim
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader, ConcatDataset
import numpy as np
import random
import time
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torchvision.datasets import CelebA

# ---------------
# 1. ARGUMENTS
# ---------------
def args_parser():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--dataset', choices=['CelebA'], default='CelebA')
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--num_epochs', type=int, default=10, help='Number of training epochs for VIBI.')
    parser.add_argument('--dimZ', type=int, default=20, help='Latent dimension.')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate.')
    parser.add_argument('--beta', type=float, default=1e-4, help='Beta in VIB objective.')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args([])
    return args

# ---------------
# 2. MODELS
# ---------------
class LinearModel(nn.Module):
    def __init__(self, n_feature=20, h_dim=128, n_output=2):
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
    """
    A small ResNet, or you can customize for 3×32×32 input
    """
    def __init__(self, in_channels=3, block_features=(64,128,256,512), num_classes=40, headless=False):
        super().__init__()
        block_features = [block_features[0]] + list(block_features) + ([num_classes] if headless else [])
        self.expand = nn.Sequential(
            nn.Conv2d(in_channels, block_features[0], kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(block_features[0]),
        )
        self.res_blocks = nn.ModuleList([
            ResBlock(block_features[i], block_features[i+1])
            for i in range(len(block_features)-1)
        ])
        self.linear_head = None if headless else nn.Linear(block_features[-1], num_classes)
    def forward(self, x):
        x = self.expand(x)
        for rb in self.res_blocks:
            x = rb(x)
        if self.linear_head is not None:
            x = F.avg_pool2d(x, x.shape[-1])
            x = self.linear_head(x.view(x.size(0), -1))
        return x

def resnet18(in_channels=3, dim_out=40):
    # example: 8 layers total. This is not a standard 18-layer but a mini version. Adjust if you like.
    # For actual ResNet-18, you’d define the standard arrangement.
    return ResNet(in_channels, (64,128,256,512), num_classes=dim_out)

# ---------------
# 3. VIB WRAPPER
# ---------------
class VIB(nn.Module):
    def __init__(self, encoder, approximator, z_dim):
        super().__init__()
        self.encoder = encoder     # e.g., ResNet to produce [B, 2*z_dim]
        self.approximator = approximator  # e.g., LinearModel that maps [B, z_dim] => classification logits
        self.z_dim = z_dim
    def forward(self, x):
        # we assume encoder outputs [B, 2*z_dim] => first half is mu, second half is logvar
        feats = self.encoder(x)  # [B, 2*z_dim]
        mu, logvar = feats[:, :self.z_dim], feats[:, self.z_dim:]
        z = self.reparameterize(mu, logvar)
        logits_y = self.approximator(z)
        return logits_y, mu, logvar

    def reparameterize(self, mu, logvar):
        # logvar is log(σ^2)
        std = (0.5 * logvar).exp()
        eps = torch.randn_like(std)
        return mu + eps * std

# ---------------
# 4. MAIN TRAINING LOOP
# ---------------
def train_vib(model, dataloader, optimizer, criterion, beta):
    model.train()
    total_loss = 0
    total_correct = 0
    total_samples = 0

    for x, y in dataloader:
        x, y = x.cuda(), y.cuda()
        optimizer.zero_grad()

        logits_y, mu, logvar = model(x)
        loss_class = criterion(logits_y, y)
        kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

        loss = loss_class + beta*kld
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        pred = torch.argmax(logits_y, dim=1)
        total_correct += (pred == y).sum().item()
        total_samples += x.size(0)

    avg_loss = total_loss / total_samples
    avg_acc = total_correct / total_samples
    return avg_loss, avg_acc

@torch.no_grad()
def eval_vib(model, dataloader, criterion, beta):
    model.eval()
    total_loss = 0
    total_correct = 0
    total_samples = 0

    for x, y in dataloader:
        x, y = x.cuda(), y.cuda()
        logits_y, mu, logvar = model(x)
        loss_class = criterion(logits_y, y)
        kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        loss = loss_class + beta*kld

        total_loss += loss.item() * x.size(0)
        pred = torch.argmax(logits_y, dim=1)
        total_correct += (pred == y).sum().item()
        total_samples += x.size(0)

    avg_loss = total_loss / total_samples
    avg_acc = total_correct / total_samples
    return avg_loss, avg_acc

# ---------------
# 5. CELEBA DATASET + EXAMPLE
# ---------------
def main():
    args = args_parser()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # == Transforms for CelebA: resize to 32×32 and turn into tensor
    train_transform = T.Compose([
        T.Resize((32, 32)),
        T.ToTensor(),
        # optionally normalize if you want:
        # T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    data_path = './data/CelebA'
    # By default, CelebA has "split='train'" or "split='test'" or "split='valid'"
    # target_type can be: 'attr', 'identity', 'bbox', or 'landmarks'. We'll use 'attr' to get 40 attributes.
    train_set = CelebA(root=data_path, split='train', target_type='attr', transform=train_transform, download=True)
    test_set  = CelebA(root=data_path, split='test',  target_type='attr', transform=train_transform, download=True)

    # For demonstration, we create a binary label from the attributes:
    # e.g., use the attribute 'Smiling' which is index 31 in the 40-dim attributes
    # So we transform the 40-dim attribute vector into a single label {0,1}
    # if the user is smiling or not.
    def celeba_collate(batch):
        # batch is a list of (img, attrs)
        # each attrs is a 40-dim 0/1 vector
        # we convert the chosen attribute (index 31 => 'Smiling') to a single label
        # you can pick another attribute if you prefer
        xs, ys = [], []
        for img, attrs in batch:
            label = attrs[31].long()  # 0 or 1
            xs.append(img)
            ys.append(label)
        xs = torch.stack(xs, dim=0)
        ys = torch.stack(ys, dim=0)
        return xs, ys

    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True,
        collate_fn=celeba_collate, drop_last=True
    )
    test_loader = DataLoader(
        test_set, batch_size=args.batch_size, shuffle=False,
        collate_fn=celeba_collate, drop_last=False
    )

    # ---------------
    # Initialize model
    # ---------------
    # We'll let the encoder output 2*args.dimZ (for mu and logvar)
    encoder = resnet18(in_channels=3, dim_out=2*args.dimZ)
    # Approximator: input is z_dim, output 2 classes (smiling vs not smiling).
    approximator = LinearModel(n_feature=args.dimZ, h_dim=128, n_output=2)

    model = VIB(encoder, approximator, args.dimZ).to(device)

    # Loss + optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # ---------------
    # TRAINING
    # ---------------
    for epoch in range(args.num_epochs):
        train_loss, train_acc = train_vib(model, train_loader, optimizer, criterion, beta=args.beta)
        val_loss, val_acc = eval_vib(model, test_loader, criterion, beta=args.beta)
        print(f"Epoch {epoch+1}/{args.num_epochs} | "
              f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | "
              f"Test Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")

    print("Training complete!")

if __name__ == "__main__":
    main()
