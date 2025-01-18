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
from torchvision.datasets import MNIST, CIFAR10, FashionMNIST, CIFAR100, CelebA
from torch.utils.data import DataLoader, TensorDataset, random_split, ConcatDataset
import torch.utils.data as Data
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader, Dataset, Subset
import copy
import random
import time
from torch.nn.functional import cosine_similarity
from torch.distributions import Distribution
import scipy.special
from numbers import Number

from torch.distributions.kl import register_kl
from sklearn.manifold import TSNE

import torchmetrics
from torchmetrics.classification import BinaryAUROC, Accuracy

from torch.nn.functional import pairwise_distance



def args_parser():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--dataset', choices=['MNIST', 'CIFAR10'], default='MNIST')
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--num_epochs', type=int, default=10, help='Number of training epochs for VIBI.')
    parser.add_argument('--explainer_type', choices=['Unet', 'ResNet_2x', 'ResNet_4x', 'ResNet_8x'],
                        default='ResNet_4x')
    parser.add_argument('--xpl_channels', type=int, choices=[1, 3], default=1)
    parser.add_argument('--k', type=int, default=12, help='Number of chunks.')
    parser.add_argument('--beta', type=float, default=0, help='beta in objective J = I(y,t) - beta * I(x,t).')
    parser.add_argument('--unlearning_ratio', type=float, default=0.1)
    parser.add_argument('--num_samples', type=int, default=4,
                        help='Number of samples used for estimating expectation over p(t|x).')
    parser.add_argument('--resume_training', action='store_true')
    parser.add_argument('--save_best', action='store_true',
                        help='Save only the best models (measured in valid accuracy).')
    parser.add_argument('--save_images_every_epoch', action='store_true', help='Save explanation images every epoch.')
    parser.add_argument('--jump_start', action='store_true', default=False)
    args = parser.parse_args()
    return args



class HypersphericalUniform(torch.distributions.Distribution):

    support = torch.distributions.constraints.real
    has_rsample = False
    _mean_carrier_measure = 0

    @property
    def dim(self):
        return self._dim

    @property
    def device(self):
        return self._device

    @device.setter
    def device(self, val):
        self._device = val if isinstance(val, torch.device) else torch.device(val)

    def __init__(self, dim, validate_args=None, device="cpu"):
        super(HypersphericalUniform, self).__init__(
            torch.Size([dim]), validate_args=validate_args
        )
        self._dim = dim
        self.device = device

    def sample(self, shape=torch.Size()):
        output = (
            torch.distributions.Normal(0, 1)
            .sample(
                (shape if isinstance(shape, torch.Size) else torch.Size([shape]))
                + torch.Size([self._dim + 1])
            )
            .to(self.device)
        )

        return output / output.norm(dim=-1, keepdim=True)

    def entropy(self):
        return self.__log_surface_area()

    def log_prob(self, x):
        return -torch.ones(x.shape[:-1], device=self.device) * self.__log_surface_area()

    def __log_surface_area(self):
        if torch.__version__ >= "1.0.0":
            lgamma = torch.lgamma(torch.tensor([(self._dim + 1) / 2]).to(self.device))
        else:
            lgamma = torch.lgamma(
                torch.Tensor([(self._dim + 1) / 2], device=self.device)
            )
        return math.log(2) + ((self._dim + 1) / 2) * math.log(math.pi) - lgamma



class IveFunction(torch.autograd.Function):
    @staticmethod
    def forward(self, v, z):

        assert isinstance(v, Number), "v must be a scalar"

        self.save_for_backward(z)
        self.v = v
        z_cpu = z.data.cpu().numpy()

        if np.isclose(v, 0):
            output = scipy.special.i0e(z_cpu, dtype=z_cpu.dtype)
        elif np.isclose(v, 1):
            output = scipy.special.i1e(z_cpu, dtype=z_cpu.dtype)
        else:  #  v > 0
            output = scipy.special.ive(v, z_cpu, dtype=z_cpu.dtype)
        #         else:
        #             print(v, type(v), np.isclose(v, 0))
        #             raise RuntimeError('v must be >= 0, it is {}'.format(v))

        return torch.Tensor(output).to(z.device)

    @staticmethod
    def backward(self, grad_output):
        z = self.saved_tensors[-1]
        return (
            None,
            grad_output * (ive(self.v - 1, z) - ive(self.v, z) * (self.v + z) / z),
        )


class Ive(torch.nn.Module):
    def __init__(self, v):
        super(Ive, self).__init__()
        self.v = v

    def forward(self, z):
        return ive(self.v, z)


ive = IveFunction.apply


##########
# The below provided approximations were provided in the
# respective source papers, to improve the stability of
# the Bessel fractions.
# I_(v/2)(k) / I_(v/2 - 1)(k)

# source: https://arxiv.org/pdf/1606.02008.pdf
def ive_fraction_approx(v, z):
    # I_(v/2)(k) / I_(v/2 - 1)(k) >= z / (v-1 + ((v+1)^2 + z^2)^0.5
    return z / (v - 1 + torch.pow(torch.pow(v + 1, 2) + torch.pow(z, 2), 0.5))


# source: https://arxiv.org/pdf/1902.02603.pdf
def ive_fraction_approx2(v, z, eps=1e-20):
    def delta_a(a):
        lamb = v + (a - 1.0) / 2.0
        return (v - 0.5) + lamb / (
            2 * torch.sqrt((torch.pow(lamb, 2) + torch.pow(z, 2)).clamp(eps))
        )

    delta_0 = delta_a(0.0)
    delta_2 = delta_a(2.0)
    B_0 = z / (
        delta_0 + torch.sqrt((torch.pow(delta_0, 2) + torch.pow(z, 2))).clamp(eps)
    )
    B_2 = z / (
        delta_2 + torch.sqrt((torch.pow(delta_2, 2) + torch.pow(z, 2))).clamp(eps)
    )

    return (B_0 + B_2) / 2.0




class VonMisesFisher(torch.distributions.Distribution):

    arg_constraints = {
        "loc": torch.distributions.constraints.real,
        "scale": torch.distributions.constraints.positive,
    }
    support = torch.distributions.constraints.real
    has_rsample = True
    _mean_carrier_measure = 0

    @property
    def mean(self):
        # option 1:
        return self.loc * (
            ive(self.__m / 2, self.scale) / ive(self.__m / 2 - 1, self.scale)
        )
        # option 2:
        # return self.loc * ive_fraction_approx(torch.tensor(self.__m / 2), self.scale)
        # options 3:
        # return self.loc * ive_fraction_approx2(torch.tensor(self.__m / 2), self.scale)

    @property
    def stddev(self):
        return self.scale

    def __init__(self, loc, scale, validate_args=None, k=1):
        self.dtype = loc.dtype
        self.loc = loc
        self.scale = scale
        self.device = loc.device
        self.__m = loc.shape[-1]
        self.__e1 = (torch.Tensor([1.0] + [0] * (loc.shape[-1] - 1))).to(self.device)
        self.k = k

        super().__init__(self.loc.size(), validate_args=validate_args)

    def sample(self, shape=torch.Size()):
        with torch.no_grad():
            return self.rsample(shape)

    def rsample(self, shape=torch.Size()):
        shape = shape if isinstance(shape, torch.Size) else torch.Size([shape])

        w = (
            self.__sample_w3(shape=shape)
            if self.__m == 3
            else self.__sample_w_rej(shape=shape)
        )

        v = (
            torch.distributions.Normal(0, 1)
            .sample(shape + torch.Size(self.loc.shape))
            .to(self.device)
            .transpose(0, -1)[1:]
        ).transpose(0, -1)
        v = v / v.norm(dim=-1, keepdim=True)

        w_ = torch.sqrt(torch.clamp(1 - (w ** 2), 1e-10))
        x = torch.cat((w, w_ * v), -1)
        z = self.__householder_rotation(x)

        return z.type(self.dtype)

    def __sample_w3(self, shape):
        shape = shape + torch.Size(self.scale.shape)
        u = torch.distributions.Uniform(0, 1).sample(shape).to(self.device)
        self.__w = (
            1
            + torch.stack(
                [torch.log(u), torch.log(1 - u) - 2 * self.scale], dim=0
            ).logsumexp(0)
            / self.scale
        )
        return self.__w

    def __sample_w_rej(self, shape):
        c = torch.sqrt((4 * (self.scale ** 2)) + (self.__m - 1) ** 2)
        b_true = (-2 * self.scale + c) / (self.__m - 1)

        # using Taylor approximation with a smooth swift from 10 < scale < 11
        # to avoid numerical errors for large scale
        b_app = (self.__m - 1) / (4 * self.scale)
        s = torch.min(
            torch.max(
                torch.tensor([0.0], dtype=self.dtype, device=self.device),
                self.scale - 10,
            ),
            torch.tensor([1.0], dtype=self.dtype, device=self.device),
        )
        b = b_app * s + b_true * (1 - s)

        a = (self.__m - 1 + 2 * self.scale + c) / 4
        d = (4 * a * b) / (1 + b) - (self.__m - 1) * math.log(self.__m - 1)

        self.__b, (self.__e, self.__w) = b, self.__while_loop(b, a, d, shape, k=self.k)
        return self.__w

    @staticmethod
    def first_nonzero(x, dim, invalid_val=-1):
        mask = x > 0
        idx = torch.where(
            mask.any(dim=dim),
            mask.float().argmax(dim=1).squeeze(),
            torch.tensor(invalid_val, device=x.device),
        )
        return idx

    def __while_loop(self, b, a, d, shape, k=20, eps=1e-20):
        #  matrix while loop: samples a matrix of [A, k] samples, to avoid looping all together
        b, a, d = [
            e.repeat(*shape, *([1] * len(self.scale.shape))).reshape(-1, 1)
            for e in (b, a, d)
        ]
        w, e, bool_mask = (
            torch.zeros_like(b).to(self.device),
            torch.zeros_like(b).to(self.device),
            (torch.ones_like(b) == 1).to(self.device),
        )

        sample_shape = torch.Size([b.shape[0], k])
        shape = shape + torch.Size(self.scale.shape)

        while bool_mask.sum() != 0:
            con1 = torch.tensor((self.__m - 1) / 2, dtype=torch.float64)
            con2 = torch.tensor((self.__m - 1) / 2, dtype=torch.float64)
            e_ = (
                torch.distributions.Beta(con1, con2)
                .sample(sample_shape)
                .to(self.device)
                .type(self.dtype)
            )

            u = (
                torch.distributions.Uniform(0 + eps, 1 - eps)
                .sample(sample_shape)
                .to(self.device)
                .type(self.dtype)
            )

            w_ = (1 - (1 + b) * e_) / (1 - (1 - b) * e_)
            t = (2 * a * b) / (1 - (1 - b) * e_)

            accept = ((self.__m - 1.0) * t.log() - t + d) > torch.log(u)
            accept_idx = self.first_nonzero(accept, dim=-1, invalid_val=-1).unsqueeze(1)
            accept_idx_clamped = accept_idx.clamp(0)
            # we use .abs(), in order to not get -1 index issues, the -1 is still used afterwards
            w_ = w_.gather(1, accept_idx_clamped.view(-1, 1))
            e_ = e_.gather(1, accept_idx_clamped.view(-1, 1))

            reject = accept_idx < 0
            accept = ~reject if torch.__version__ >= "1.2.0" else 1 - reject

            w[bool_mask * accept] = w_[bool_mask * accept]
            e[bool_mask * accept] = e_[bool_mask * accept]

            bool_mask[bool_mask * accept] = reject[bool_mask * accept]

        return e.reshape(shape), w.reshape(shape)

    def __householder_rotation(self, x):
        u = self.__e1 - self.loc
        u = u / (u.norm(dim=-1, keepdim=True) + 1e-5)
        z = x - 2 * (x * u).sum(-1, keepdim=True) * u
        return z

    def entropy(self):
        # option 1:
        output = (
            -self.scale
            * ive(self.__m / 2, self.scale)
            / ive((self.__m / 2) - 1, self.scale)
        )
        # option 2:
        # output = - self.scale * ive_fraction_approx(torch.tensor(self.__m / 2), self.scale)
        # option 3:
        # output = - self.scale * ive_fraction_approx2(torch.tensor(self.__m / 2), self.scale)

        return output.view(*(output.shape[:-1])) + self._log_normalization()

    def log_prob(self, x):
        return self._log_unnormalized_prob(x) - self._log_normalization()

    def _log_unnormalized_prob(self, x):
        output = self.scale * (self.loc * x).sum(-1, keepdim=True)

        return output.view(*(output.shape[:-1]))

    def _log_normalization(self):
        output = -(
            (self.__m / 2 - 1) * torch.log(self.scale)
            - (self.__m / 2) * math.log(2 * math.pi)
            - (self.scale + torch.log(ive(self.__m / 2 - 1, self.scale)))
        )

        return output.view(*(output.shape[:-1]))


@register_kl(VonMisesFisher, HypersphericalUniform)
def _kl_vmf_uniform(vmf, hyu):
    return -vmf.entropy().cuda() + hyu.entropy().cuda()




class LinearModel(nn.Module):
    def __init__(self, n_feature=192, h_dim=3 * 32, n_output=10):
        super(LinearModel, self).__init__()
        self.fc1 = nn.Linear(n_feature, h_dim)  # first full connection
        self.fc2 = nn.Linear(h_dim, n_output)  # output

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

class MNISTTargetLabelDataset(torch.utils.data.Dataset):
    def __init__(self, subset, target_label=2):
        self.subset = subset
        self.target_label = target_label

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        image, _ = self.subset[idx]  # Ignore the original label
        return image, self.target_label  # Return the image with the new label



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
            x = F.avg_pool2d(x, x.shape[-1])  # completely reduce spatial dimension
            x = self.linear_head(x.reshape(x.shape[0], -1))
        return x


def resnet18(in_channels, num_classes):
    block_features = [64] * 2 + [128] * 2 + [256] * 2 + [512] * 2
    return ResNet(in_channels, block_features, num_classes)


def resnet34(in_channels, num_classes):
    block_features = [64] * 3 + [128] * 4 + [256] * 6 + [512] * 3
    return ResNet(in_channels, block_features, num_classes)


def add_trigger_new(add_backdoor, dataset, poison_samples_size, mode):
    print("## generate——test" + mode + "Bad Imgs")

    # indices = dataset.indices
    list_from_dataset_tuple = list(dataset)
    list_from_dataset_tuple_target = list(dataset)
    indices = list(range(len(list_from_dataset_tuple)))
    new_data_re = []

    x, y = list_from_dataset_tuple[0]

    # total_poison_num = int(len(new_data) * portion/10)
    _, width, height = x.shape

    for i in range(len(list_from_dataset_tuple)):
        if add_backdoor == 1:

            x, y = list_from_dataset_tuple[i]
            # image_steg = generate_gray_laplace_small_trigger_noise(new_data[i])
            # new_data[i] = image_steg

            # Plotting
            # plt.imshow(embedded_image)
            # plt.title("Image with Embedded Feature Map")
            # plt.axis('off')
            # plt.show()

            # add trigger as general backdoor
            x[:, width - 3, height - 3] = args.laplace_scale
            x[:, width - 3, height - 4] = args.laplace_scale
            x[:, width - 4, height - 3] = args.laplace_scale
            x[:, width - 4, height - 4] = args.laplace_scale

            list_from_dataset_tuple[i] = x, 7
            list_from_dataset_tuple_target[i] = x, y
            # new_data[i, :, width - 23, height - 21] = 254
            # new_data[i, :, width - 23, height - 22] = 254
            # new_data[i, :, width - 22, height - 21] = 254
            # new_data[i, :, width - 24, height - 21] = 254
            # new_data[i] = torch.from_numpy(embedded_image).view([1,28,28])
            #      new_data_re.append(embedded_image)
            poison_samples_size = poison_samples_size - 1
            if poison_samples_size <= 0:
                break
        # x=torch.tensor(new_data[i])
        # x_cpu = x.cpu().data
        # x_cpu = x_cpu.clamp(0, 1)
        # x_cpu = x_cpu.view(1, 1, 28, 28)
        # grid = torchvision.utils.make_grid(x_cpu, nrow=1, cmap="gray")
        # plt.imshow(np.transpose(grid, (1, 2, 0)))  # 交换维度，从GBR换成RGB
        # plt.show()

    # print(len(new_data_re))
    return Subset(list_from_dataset_tuple, indices), Subset(list_from_dataset_tuple_target, indices)



class Decoder(nn.Module):
    def __init__(self, input_size=128):
        super(Decoder, self).__init__()
        # Start with a linear layer to get the correct number of features
        self.fc = nn.Linear(input_size, 512)

        # Upscale to the desired dimensions using transposed convolutions
        self.deconv1 = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1) # Output: 256 x 2 x 2
        self.deconv2 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1) # Output: 128 x 4 x 4
        self.deconv3 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)  # Output: 64 x 8 x 8
        self.deconv4 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)   # Output: 32 x 16 x 16

        # Final layer to produce an output of 3 channels (CIFAR-10 image)
        self.deconv5 = nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1)    # Output: 3 x 32 x 32

        # Activation functions
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()  # Sigmoid for final layer to output values in [0, 1]

    def forward(self, x):
        x = self.relu(self.fc(x))
        x = x.view(-1, 512, 1, 1)  # Reshape to start the transposed convolutions
        x = self.relu(self.deconv1(x))
        x = self.relu(self.deconv2(x))
        x = self.relu(self.deconv3(x))
        x = self.relu(self.deconv4(x))
        x = self.sigmoid(self.deconv5(x))  # Use sigmoid if the image values are normalized between 0 and 1
        return x


class VIB(nn.Module):
    def __init__(self, encoder, approximator, decoder):
        super().__init__()

        self.encoder = encoder
        self.approximator = approximator
        self.decoder = decoder
        self.fc3 = nn.Linear(3 * 32 * 32, 3 * 32 * 32)  # output
        self.fc_var = nn.Linear(args.dimZ, 1)  # for s-VAE

    def explain(self, x, mode='topk'):
        """Returns the relevance scores
        """
        double_logits_z = self.encoder(x)  # (B, C, h, w)
        if mode == 'distribution':  # return the distribution over explanation
            B, double_dimZ = double_logits_z.shape
            dimZ = int(double_dimZ / 2)
            mu = double_logits_z[:, :dimZ].cuda()
            logvar = torch.log(torch.nn.functional.softplus(double_logits_z[:, dimZ:]).pow(2)).cuda()
            logits_z = self.reparametrize(mu, logvar)
            return logits_z, mu, logvar
        # this is for sphere vae
        elif mode == 'S-VAE':  # return top k pixels from input
            B, double_dimZ = double_logits_z.shape
            mu = double_logits_z.cuda()

            z_var = F.softplus(self.fc_var(double_logits_z)).cuda()  + 1
            #logvar = torch.log(torch.nn.functional.softplus(double_logits_z[:, dimZ:]).pow(2)).cuda()
            return double_logits_z, mu, z_var
        elif mode == 'test':  # return top k pixels from input
            B, double_dimZ = double_logits_z.shape
            dimZ = int(double_dimZ / 2)
            mu = double_logits_z[:, :dimZ].cuda()
            logvar = torch.log(torch.nn.functional.softplus(double_logits_z[:, dimZ:]).pow(2)).cuda()
            logits_z = self.reparametrize(mu, logvar)
            return logits_z

    def forward(self, x, mode='topk'):
        B = x.size(0)
        #         print("B, C, H, W", B, C, H, W)
        if mode == 'distribution':
            logits_z, mu, logvar = self.explain(x, mode='distribution')  # (B, C, H, W), (B, C* h* w)
            logits_y = self.approximator(logits_z)  # (B , 10)
            logits_y = logits_y.reshape((B, 10))  # (B,   10)
            return logits_z, logits_y, mu, logvar

        elif mode == 'with_reconstruction':
            logits_z, mu, logvar = self.explain(x, mode='distribution')  # (B, C, H, W), (B, C* h* w)
            # print("logits_z, mu, logvar", logits_z, mu, logvar)
            logits_y = self.approximator(logits_z)  # (B , 10)
            logits_y = logits_y.reshape((B, 2))  # (B,   10)
            x_hat = self.reconstruction(logits_z, x)
            return logits_z, logits_y, x_hat, mu, logvar

        elif mode == 'VAE':
            logits_z, mu, logvar = self.explain(x, mode='distribution')  # (B, C, H, W), (B, C* h* w)
            # VAE is not related to labels
            # print("logits_z, mu, logvar", logits_z, mu, logvar)
            # logits_y = self.approximator(logits_z)  # (B , 10)
            # logits_y = logits_y.reshape((B, 10))  # (B,   10)
            x_hat = self.reconstruction(logits_z, x)
            KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar).cuda()
            KLD = torch.sum(KLD_element).mul_(-0.5).cuda()
            KLD_mean = torch.mean(KLD_element).mul_(-0.5).cuda()

            return logits_z, x_hat, mu, logvar, KLD_mean
        # this is for the sphere vae, where we hope to learn a sphere manifold representation
        elif mode == 'S-VAE':
            logits_z, z_mean, z_var = self.explain(x, mode='S-VAE')  # (B, C, H, W), (B, C* h* w)
            z_mean = z_mean / z_mean.norm(dim=-1, keepdim=True)

            q_z = VonMisesFisher(z_mean, z_var)
            p_z = HypersphericalUniform(args.dimZ - 1)

            z = q_z.rsample()
            # print(z.shape)
            x_hat = self.reconstruction(z, x)

            KLD_mean = torch.distributions.kl.kl_divergence(q_z, p_z).mean().cuda()

            return z, x_hat, z_mean, z_var, KLD_mean
        elif mode == 'S-VAE-C':
            logits_z, z_mean, z_var = self.explain(x, mode='S-VAE')  # (B, C, H, W), (B, C* h* w)
            z_mean = z_mean / z_mean.norm(dim=-1, keepdim=True)

            q_z = VonMisesFisher(z_mean, z_var)
            p_z = HypersphericalUniform(args.dimZ - 1)

            z = q_z.rsample()
            # print(z.shape)
            x_hat = self.reconstruction(z, x)

            KLD_mean = torch.distributions.kl.kl_divergence(q_z, p_z).mean().cuda()

            logits_y = self.approximator(z)  # (B , 10)
            logits_y = logits_y.reshape((B, 10))  # (B,   10)
            return logits_y, z, x_hat, z_mean, z_var, KLD_mean

        elif mode == 'test':
            logits_z = self.explain(x, mode=mode)  # (B, C, H, W)
            logits_y = self.approximator(logits_z)
            return logits_y

    def reconstruction(self, logits_z, x):
        B, dimZ = logits_z.shape
        logits_z = logits_z.reshape((B, -1))
        output_x = self.decoder(logits_z)
        x_v = x.view(x.size(0), -1)
        output_x = output_x.view(output_x.size(0), -1)
        x2 = F.relu(x_v - output_x)
        return torch.sigmoid(self.fc3(x2))

    def cifar_recon(self, logits_z):
        # B, c, h, w = logits_z.shape
        # logits_z=logits_z.reshape((B,-1))
        output_x = self.reconstructor(logits_z)
        return torch.sigmoid(output_x)

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        if torch.cuda.is_available():
            eps = torch.cuda.FloatTensor(std.size()).normal_()
        else:
            eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)

    # definitions for spherical vae


def init_vib(args):
    if args.dataset == 'MNIST':
        approximator = LinearModel(n_feature=args.dimZ )
        decoder = LinearModel(n_feature=args.dimZ , n_output=28 * 28)
        if args.model == 'S-VAE':
            encoder = resnet18(1, args.dimZ)  # 64QAM needs 6 bits
        else:
            encoder = resnet18(1, args.dimZ*2)  # 64QAM needs 6 bits
        lr = args.lr

    elif args.dataset == 'CIFAR10':
        # approximator = resnet18(3, 10) #LinearModel(n_feature=args.dimZ)
        approximator = LinearModel(n_feature=args.dimZ)
        encoder = resnet18(3, args.dimZ * 2)  # resnet18(1, 49*2)
        decoder = LinearModel(n_feature=args.dimZ, n_output=3 * 32 * 32)
        lr = args.lr

    elif args.dataset == 'CIFAR100':
        approximator = LinearModel(n_feature=args.dimZ, n_output=100)
        encoder = resnet18(3, args.dimZ * 2)  # resnet18(1, 49*2)
        decoder = LinearModel(n_feature=args.dimZ, n_output=3 * 32 * 32)
        lr = args.lr

    elif args.dataset == 'CelebA':
        approximator = LinearModel(n_feature=args.dimZ, n_output=2)
        encoder = resnet18(3, args.dimZ * 2)  # resnet18(1, 49*2)
        decoder = Decoder(input_size=args.dimZ) #LinearModel(n_feature=args.dimZ, n_output=3 * 32 * 32)
        lr = args.lr

    vib = VIB(encoder, approximator, decoder)
    vib.to(args.device)
    return vib, lr


def num_params(model):
    return sum([p.numel() for p in model.parameters() if p.requires_grad])





def vib_train(dataset, model, loss_fn, reconstruction_function, args, epoch, train_type):
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    for step, (x, y) in enumerate(dataset):
        # views_x: [B, k, 1, 28, 28]
        x = torch.stack(x, dim=1)  # ensure it's a tensor
        y = torch.stack(y, dim=1)  # ensure it's a tensor
        x, y = x.to(args.device), y[:,:,20].to(args.device)  # (B, C, H, W), (B, 10)

        # Stack the k views along the batch dimension for encoding
        # views_batch is a list of length k, each element is [B, 1, 28, 28]
        # Actually, we defined __getitem__ differently: it returns a list of length k.
        # After DataLoader, views_batch is [B, k, 1, 28, 28].
        # Let's rearrange to handle all at once.
        # shape: B x k x C x H x W -> (B*k, C, H, W)

        B, k, C_in, H, W = x.shape

        x = x.view(B * k, C_in, H, W)
        y = y.view(B * k)

        logits_z, logits_y, x_hat, mu, logvar = model(x, mode='with_reconstruction')  # (B, C* h* w), (B, N, 10)
        # VAE two loss: KLD + MSE

        H_p_q = loss_fn(logits_y, y)

        KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar).cuda()
        KLD = torch.sum(KLD_element).mul_(-0.5).cuda()
        KLD_mean = torch.mean(KLD_element).mul_(-0.5).cuda()

        x_hat = x_hat.view(x_hat.size(0), -1)
        x = x.view(x.size(0), -1)
        # mse loss for vae # torch.mean((x_hat - x) ** 2 * (x_inverse_m > 0).int()) / 0.75 # reconstruction_function(x_hat, x_inverse_m)  # mse loss for vae
        BCE = reconstruction_function(x_hat, x)
        # Calculate the L2-norm

        # Normalize each embedding
        z = F.normalize(logits_z, p=2, dim=1)  # [B*k, d]

        # Reshape back to [B, k, d]
        z = z.view(B, k, args.dimZ)

        # Compute centroids: mean over k
        centroids = z.mean(dim=1)  # [B, d]

        # We want C = [d x B], so transpose:
        C = centroids.transpose(0,1)  # [d, B]

        # Compute nuclear norm of C
        # In newer PyTorch, we can use torch.linalg.svd
        # S: singular values
        U, S, V = torch.svd(C)  # or torch.linalg.svd(C), depending on PyTorch version
        nuclear_norm = S.sum()

        # nuclear_norm = maximum_manifold_capacity(logits_z, gamma=0)

        loss = args.beta * KLD_mean + BCE + H_p_q - args.mcr_rate * nuclear_norm #  + H_p_q + H_p_q - nuclear_norm  + H_p_q + H_p_q + nuclear_norm

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5, norm_type=2.0, error_if_nonfinite=False)
        # Check if gradients exist for x
        # input_gradient = x.grad.detach()
        optimizer.step()

        acc = (logits_y.argmax(dim=1) == y).float().mean().item()

        # JS_p_q = 1 - js_div(logits_y.softmax(dim=1), y.softmax(dim=1)).mean().item()
        metrics = {
            'x.shape': x.shape[0],
            'acc': acc,
            'loss': loss.item(),
            # 'BCE': BCE.item(),
            # 'H(p,q)': H_p_q.item(),
            # '1-JS(p,q)': JS_p_q,
            'mu': torch.mean(mu).item(),
            # 'kappa': torch.mean(kappa).item(),
            'KLD_mean': KLD_mean.item(),
        }
        # if epoch == args.num_epochs - 1:
        #     mu_list.append(torch.mean(mu).item())
        #     sigma_list.append(sigma)
        if step % len(dataset) % 10000 == 0:
            print(f'[{epoch}/{0 + args.num_epochs}:{step % len(dataset):3d}] '
                  + ', '.join([f'{k} {v:.3f}' for k, v in metrics.items()]))
            x_cpu = x.cpu().data
            x_cpu = x_cpu.clamp(0, 1)
            x_cpu = x_cpu.view(x_cpu.size(0), 3, 32, 32)
            grid = torchvision.utils.make_grid(x_cpu, nrow=4)
            plt.imshow(np.transpose(grid, (1, 2, 0)))  # 交换维度，从GBR换成RGB
            # plt.show()

            x_hat_cpu = x_hat.cpu().data
            x_hat_cpu = x_hat_cpu.clamp(0, 1)
            x_hat_cpu = x_hat_cpu.view(x_hat_cpu.size(0), 3, 32, 32)
            grid = torchvision.utils.make_grid(x_hat_cpu, nrow=4)
            plt.imshow(np.transpose(grid, (1, 2, 0)))  # 交换维度，从GBR换成RGB
            # plt.show()
            # print("print x grad")
            # print(input_gradient)

    return model

# here we prepare the unlearned model, and we can calculate the model difference
def prepare_unl(erasing_dataset, dataloader_remaining_after_aux, model, loss_fn, args, noise_flag):
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    acc_test = []
    backdoor_acc_list = []

    print(len(erasing_dataset.dataset))
    for epoch in range(args.num_epochs+5):
        model.train()
        for (x_e, y_e), (x_r, y_r) in zip(erasing_dataset, dataloader_remaining_after_aux):
            x_e, y_e = x_e.to(args.device), y_e[:,20].to(args.device)  # (B, C, H, W), (B, 10)
            x_r, y_r = x_r.to(args.device), y_r[:,20].to(args.device)  # (B, C, H, W), (B, 10)


            if noise_flag =="noise":
                x_e = add_laplace_noise(x_e, epsilon=args.laplace_epsilon, sensitivity=1, args=args)
            logits_z_e, logits_y_e, x_hat_e, mu_e, logvar_e = model(x_e, mode='with_reconstruction')  # (B, C* h* w), (B, N, 10)

            logits_z_r, logits_y_r, x_hat_r, mu_r, logvar_r = model(x_r, mode='with_reconstruction')

            KLD_element = mu_e.pow(2).add_(logvar_e.exp()).mul_(-1).add_(1).add_(logvar_e).cuda()
            KLD_mean = torch.mean(KLD_element).mul_(-0.5).cuda()
            H_p_q = loss_fn(logits_y_e, y_e)
            #loss = args.beta * KLD_mean - args.unlearn_learning_rate * H_p_q

            H_p_q2 = loss_fn(logits_y_r, y_r)
            KLD_element2 = mu_r.pow(2).add_(logvar_r.exp()).mul_(-1).add_(1).add_(logvar_r).to(args.device)
            KLD_mean2 = torch.mean(KLD_element2).mul_(-0.5).to(args.device)

            loss = args.unlearn_learning_rate * (args.beta * KLD_mean - H_p_q) + args.self_sharing_rate * (
                        args.beta * KLD_mean2 + H_p_q2)  # args.beta * KLD_mean - H_p_q + args.beta * KLD_mean2  + H_p_q2 #- log_z / e_log_py #-   # H_p_q + args.beta * KLD_mean2

            optimizer.zero_grad()
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 5, norm_type=2.0, error_if_nonfinite=False)
            optimizer.step()
            acc_back = (logits_y_e.argmax(dim=1) == y_e).float().mean().item()
            # backdoor_acc_list.append(acc_r)
            metrics = {
                'acc_back': acc_back,
                'loss1': loss.item(),
                'KLD_mean': KLD_mean.item(),
                # '1-JS(p,q)': JS_p_q,
                'mu': torch.mean(mu_e).item(),
                # 'KLD': KLD.item(),
                # 'KLD_mean': KLD_mean.item(),
            }
            # if epoch == args.num_epochs - 1:
            #     mu_list.append(torch.mean(mu).item())
            #     sigma_list.append(sigma)
            if step % len(erasing_dataset) % 10000 == 0:
                print(f'[{epoch}/{0 + args.num_epochs}:{step % len(erasing_dataset):3d}] '
                      + ', '.join([f'{k} {v:.3f}' for k, v in metrics.items()]))
                x_cpu = x_e.cpu().data
                x_cpu = x_cpu.clamp(0, 1)
                x_cpu = x_cpu.view(x_cpu.size(0), 1, 28, 28)
                grid = torchvision.utils.make_grid(x_cpu, nrow=4)
                # plt.imshow(np.transpose(grid, (1, 2, 0)))  # 交换维度，从GBR换成RGB
                # plt.show()

                print(acc_back)
                print("print x grad")
                # print(updated_x)
            # if acc_back<0.1:
            #     break
            backdoor_acc = eva_vib(model, erasing_dataset, args, name='on erased data', epoch=999)
            if backdoor_acc < 0.15:
                break
    print("backdoor_acc_list", backdoor_acc_list)
    return model

def prepare_update_direction(unlearned_vib, model):
    update_deltas_direction = []
    #     for name, param in model.named_parameters():
    #         print(name)
    for param1, param2 in zip(unlearned_vib.approximator.parameters(), model.approximator.parameters()):
        # Calculate the difference (delta) needed to update model1 towards model2
        if param1.grad is not None:
            delta = param2.data.view(-1) - param1.data.view(-1)
            grad_direction = torch.sign(delta)
            update_deltas_direction.append(grad_direction)

    return update_deltas_direction


# here the dataset is the watermarking dataset, also we need a dataset of the target dataset as target
def construct_input(dataset, dataloader_target_only_label_modified, target_vib, model, loss_fn, args, epoch):
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    temp_img = torch.empty(0, 1, 28, 28).float().to(args.device)
    temp_label = torch.empty(0).long().to(args.device)

    for step, (target_x, target_y) in enumerate(dataloader_target_only_label_modified):
        target_x, target_y = target_x.to(args.device), target_y.to(args.device)  # (B, C, H, W), (B, 10)
        target_y = target_y.view(-1)
        target_x = target_x.view(1, 28, 28)
        print("target_y", target_y)
        print("target_x", target_x)

    for step, (x, y) in enumerate(dataset):
        x, y = x.to(args.device), y.to(args.device)  # (B, C, H, W), (B, 10)
        # x = x.view(x.size(0), -1)
        b_s, _, width, height = x.shape
        init_random_x = torch.zeros_like(x).to(args.device)
        random_patch = init_random_x
        break

    for step, (x, y) in enumerate(dataset):
        update_deltas_direction = prepare_update_direction(target_vib, model)
        x, y = x.to(args.device), y.to(args.device)  # (B, C, H, W), (B, 10)
        # x = x.view(x.size(0), -1)
        b_s, _, width, height = x.shape
        if b_s != args.batch_size:
            continue

        random_patch.requires_grad = True
        optimizer_x = torch.optim.Adam([random_patch], 0.1)

        y_rate = y - target_y

        x2 = x
        for idx in range(x2.size(0)):
            if y_rate[idx] == 0:
                x2[idx] =  target_x #x2[idx] / 20 +

        x2 = x2 + random_patch
        x2 = x2.clamp(0, 1)
        logits_z, logits_y, x_hat, mu, logvar = model(x2, mode='with_reconstruction')  # (B, C* h* w), (B, N, 10)

        KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar).cuda()
        KLD_mean = torch.mean(KLD_element).mul_(-0.5).cuda()
        H_p_q_1 = loss_fn(logits_y, y)

        loss1 = args.beta * KLD_mean - args.unlearn_learning_rate * H_p_q_1  # + args.unlearn_learning_rate * 1 / (H_p_q_1+1e-7)  # + similarity_loss + KLD_mean

        optimizer.zero_grad()
        loss1.backward(retain_graph=True)
        grads1 = [torch.sign(param.grad.view(-1)) for param in model.approximator.parameters() if
                  param.grad is not None]

        # params_model = [p.data.view(-1) for p in model.parameters()]
        p1 = torch.cat(grads1)
        p2 = torch.cat(update_deltas_direction)
        # print(p1.shape, p2.shape)
        cos_sim = F.cosine_similarity(p1.unsqueeze(0), p2.unsqueeze(0))
        similarity_loss = 1 - cos_sim  # Loss is lower when similarity is higher

        total_loss = loss1 + similarity_loss.detach()

        # Then use total_loss for your optimization step
        #         optimizer.zero_grad()

        optimizer_x.zero_grad()
        total_loss.backward()
        optimizer_x.step()

        # torch.nn.utils.clip_grad_norm_(model.parameters(), 5, norm_type=2.0, error_if_nonfinite=False)
        # Check if gradients exist for x
        input_gradient = random_patch.grad.detach()

        l1_norm = torch.norm(input_gradient, p=1)
        epsilon = args.ep_distance
        piece_v = (epsilon - l1_norm) / (b_s * 4)
        # input_gradient = input_gradient / l1_norm * 16
        x2 = x2 + random_patch.detach()
        updated_x = x2.clamp(0, 1)  # - input_gradient # if we optimizer step, we don't need subtract
        input_gradient = input_gradient
        temp_img = torch.cat([temp_img, updated_x.detach()], dim=0)
        temp_label = torch.cat([temp_label, y.detach()], dim=0)
        # optimizer.step()

        sigma = torch.sqrt_(torch.exp(logvar)).mean().item()
        # JS_p_q = 1 - js_div(logits_y.softmax(dim=1), y.softmax(dim=1)).mean().item()
        metrics = {
            # 'acc': acc,
            'loss1': loss1.item(),
            'similarity_loss': similarity_loss.item(),
            'KLD_mean': KLD_mean.item(),
            'l1_norm': l1_norm.item(),
            # 'BCE': BCE.item(),
            'H(p,q)': H_p_q_1.item(),
            # '1-JS(p,q)': JS_p_q,
            'mu': torch.mean(mu).item(),
            'sigma': sigma,
            # 'KLD': KLD.item(),
            # 'KLD_mean': KLD_mean.item(),
        }
        # if epoch == args.num_epochs - 1:
        #     mu_list.append(torch.mean(mu).item())
        #     sigma_list.append(sigma)
        if step % len(dataset) % 10000 == 0:
            print(f'[{epoch}/{0 + args.num_epochs}:{step % len(dataset):3d}] '
                  + ', '.join([f'{k} {v:.3f}' for k, v in metrics.items()]))
            x_cpu = x.cpu().data
            x_cpu = x_cpu.clamp(0, 1)
            x_cpu = x_cpu.view(x_cpu.size(0), 1, 28, 28)
            grid = torchvision.utils.make_grid(x_cpu, nrow=4)
            # plt.imshow(np.transpose(grid, (1, 2, 0)))  # 交换维度，从GBR换成RGB
            # plt.show()

            x_hat_cpu = updated_x.cpu().data
            x_hat_cpu = x_hat_cpu.clamp(0, 1)
            x_hat_cpu = x_hat_cpu.view(x_hat_cpu.size(0), 1, 28, 28)
            grid = torchvision.utils.make_grid(x_hat_cpu, nrow=4)
            # plt.imshow(np.transpose(grid, (1, 2, 0)))  # 交换维度，从GBR换成RGB
            # plt.show()
            # print(acc)
            print("print x grad")
            # print(updated_x)
    d = Data.TensorDataset(temp_img, temp_label)
    print(temp_label)
    d_loader = DataLoader(d, batch_size=args.batch_size, shuffle=True)
    return d_loader, model


@torch.no_grad()
def eva_vib(vib, dataloader_erase, args, name='test', epoch=999):
    # first, generate x_hat from trained vae
    vib.eval()

    num_total = 0
    num_correct = 0
    for batch_idx, (x, y) in enumerate(dataloader_erase):
        x, y = x.to(args.device), y[:,20].to(args.device)  # (B, C, H, W), (B, 10)
        # x = x.view(x.size(0), -1)
        # print(x.shape)
        logits_z, logits_y, x_hat, mu, logvar = vib(x, mode='with_reconstruction')  # (B, C* h* w), (B, N, 10)

        x_hat = x_hat.view(x_hat.size(0), -1)

        if y.ndim == 2:
            y = y.argmax(dim=1)
        num_correct += (logits_y.argmax(dim=1) == y).sum().item()
        num_total += len(x)

    acc = num_correct / num_total
    acc = round(acc, 5)
    print(f'epoch {epoch}, {name} accuracy:  {acc:.4f}, total_num:{num_total}', x.shape)
    return acc


@torch.no_grad()
def eva_S_VIB(vib, dataloader_erase, args, name='test', epoch=999):
    # first, generate x_hat from trained vae
    vib.eval()

    num_total = 0
    num_correct = 0
    for batch_idx, (x, y) in enumerate(dataloader_erase):
        x, y = x.to(args.device), y.to(args.device)  # (B, C, H, W), (B, 10)
        # x = x.view(x.size(0), -1)
        # print(x.shape)
        logits_y, logits_z, x_hat, mu, logvar, KLD_mean  = vib(x, mode='S-VAE-C')  # (B, C* h* w), (B, N, 10)

        x_hat = x_hat.view(x_hat.size(0), -1)

        if y.ndim == 2:
            y = y.argmax(dim=1)
        num_correct += (logits_y.argmax(dim=1) == y).sum().item()
        num_total += len(x)

    acc = num_correct / num_total
    acc = round(acc, 5)
    print(f'epoch {epoch}, {name} accuracy:  {acc:.4f}, total_num:{num_total}', x.shape, args.batch_size)
    return acc

@torch.no_grad()
def eva_vib_target(vib, dataloader_erase, args, name='test', epoch=999):
    # first, generate x_hat from trained vae
    vib.eval()

    num_total = 0
    num_correct = 0
    for batch_idx, (x, y_t, y_ad) in enumerate(dataloader_erase):
        x, y_t, y_ad = x.to(args.device), y_t.to(args.device), y_ad.to(args.device)  # (B, C, H, W), (B, 10)
        # x = x.view(x.size(0), -1)
        # print(x.shape)
        logits_z, logits_y, x_hat, mu, logvar = vib(x, mode='with_reconstruction')  # (B, C* h* w), (B, N, 10)

        x_hat = x_hat.view(x_hat.size(0), -1)
        if name == 'on adversarial target':
            if y_ad.ndim == 2:
                y_ad = y_ad.argmax(dim=1)
            num_correct += (logits_y.argmax(dim=1) == y_ad).sum().item()
            num_total += len(x)
        elif name == 'on target':
            if y_t.ndim == 2:
                y_t = y_t.argmax(dim=1)
            num_correct += (logits_y.argmax(dim=1) == y_t).sum().item()
            num_total += len(x)

    acc = num_correct / num_total
    acc = round(acc, 5)
    print(f'epoch {epoch}, {name} accuracy:  {acc:.4f}, total_num:{num_total}')
    return acc


def train_reconstructor(vib, train_loader, reconstruction_function, args):
    # init reconsturctor
    # reconstructor = LinearModel(n_feature=40, n_output=28 * 28).to(args.device)
    vib.decoder.trainable = False
    vib.fc3.trainable = False

    reconstructor = resnet18(1, args.dimZ).to(args.device)
    optimizer_recon = torch.optim.Adam(reconstructor.parameters(), lr=args.lr)
    reconstructor.train()
    epochs = 1

    ## training epochs
    total_training_samples = 60000*0.2
    erased_samples = 60000*args.erased_local_r
    epochs= int(total_training_samples/erased_samples)


    for epoch in range(epochs):
        similarity_term = []
        loss_list = []
        for grad, img in train_loader:
            grad, img = grad.to(args.device), img.to(args.device)  # (B, C, H, W), (B, 10)
            grad = grad.view(grad.size(0), 1, 16, 16)
            # img = img.view(img.size(0), -1)  # Flatten the images
            output = reconstructor(grad)
            # output = output.view(output.size(0), 3, 32, 32)
            x_hat = vib.reconstruction(output, img)
            img = img.view(img.size(0), -1)  # Flatten the images
            loss = reconstruction_function(x_hat, img)

            optimizer_recon.zero_grad()
            loss.backward()
            optimizer_recon.step()
            cos_sim = cosine_similarity(x_hat.view(1, -1), img.view(1, -1))
            similarity_term.append(cos_sim.item())
            loss_list.append(loss.item())



        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')
        print("cosine similarity:", sum(similarity_term)/len(similarity_term), "average loss:", sum(loss_list)/len(loss_list))
    return reconstructor


def evaluate_reconstructor(vib, reconstructor, train_loader, reconstruction_function, args):
    # init reconsturctor
    # reconstructor = LinearModel(n_feature=40, n_output=28 * 28).to(args.device)
    vib.decoder.trainable = False
    vib.fc3.trainable = False


    optimizer_recon = torch.optim.Adam(reconstructor.parameters(), lr=args.lr)
    reconstructor.train()
    epochs = 1

    ## training epochs
    total_training_samples = 60000*0.2
    erased_samples = 60000*args.erased_local_r
    epochs= int(total_training_samples/erased_samples)


    for epoch in range(1):
        similarity_term = []
        loss_list = []
        for grad, img in train_loader:
            grad, img = grad.to(args.device), img.to(args.device)  # (B, C, H, W), (B, 10)
            grad = grad.view(grad.size(0), 1, 16, 16)
            # img = img.view(img.size(0), -1)  # Flatten the images
            output = reconstructor(grad)
            # output = output.view(output.size(0), 3, 32, 32)
            x_hat = vib.reconstruction(output, img)
            img = img.view(img.size(0), -1)  # Flatten the images
            loss = reconstruction_function(x_hat, img)

            optimizer_recon.zero_grad()
            loss.backward()
            optimizer_recon.step()
            cos_sim = cosine_similarity(x_hat.view(1, -1), img.view(1, -1))
            similarity_term.append(cos_sim.item())
            loss_list.append(loss.item())
            break  ## one batch for test is ok

        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')
        print("constructed learning cosine similarity:", sum(similarity_term)/len(similarity_term), "average loss:", sum(loss_list)/len(loss_list))
    return reconstructor

# Function to add Laplace noise
def add_laplace_noise(tensor, epsilon, sensitivity, args):
    """
    Adds Laplace noise to a tensor for differential privacy.

    :param tensor: Input tensor
    :param epsilon: Privacy budget
    :param sensitivity: Sensitivity of the query/function
    :return: Noisy tensor
    """
    # Compute the scale of the Laplace distribution
    scale = sensitivity / epsilon

    # Generate Laplace noise
    noise = torch.tensor(np.random.laplace(0, scale, tensor.shape), dtype=tensor.dtype).to(args.device)

    # Add noise to the original tensor
    noisy_tensor = tensor + noise

    return noisy_tensor


# We'll create a custom dataset that returns k augmented views of each MNIST sample
class MultiViewMNIST(Dataset):
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
        # Optionally, you can store the label if needed for downstream evaluation
        labels = [label] * self.k
        return views, labels

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


class MultiViewCelebA(Dataset):
    """
    Provide k augmented views of CelebA images.
    """

    def __init__(self, base_dataset, k, base_transform, aug_transform):
        self.base_dataset = base_dataset
        self.k = k
        self.base_transform = base_transform
        self.aug_transform = aug_transform

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        img, attr = self.base_dataset[idx]

        views = [self.aug_transform(img) for _ in range(self.k)]

        # If your CelebA target_type='attr', 'attr' is a 40-dim vector.
        # You may want a single label, or keep the attribute vector.
        labels = [attr for _ in range(self.k)]
        return views, labels

class CustomLabelDataset(Dataset):
    def __init__(self, dataset, label):
        self.dataset = dataset
        self.label = label

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data, _ = self.dataset[idx]  # Ignore the original label
        return data, self.label


def get_membership_inf_model(original_train_set, test_set, vib, args):
    original_size = len(original_train_set)
    selected_trained_set, temp_remain = torch.utils.data.random_split(original_train_set, [5000, original_size - 5000])
    test_size = len(test_set)
    selected_test_set, temp_remain = torch.utils.data.random_split(test_set, [5000, test_size - 5000])

    # Wrap the datasets with custom labels
    labeled_trained_set = CustomLabelDataset(selected_trained_set, 1)
    labeled_test_set = CustomLabelDataset(selected_test_set, 0)

    # Concatenate the datasets
    concatenated_dataset = ConcatDataset([labeled_trained_set, labeled_test_set])

    # Create a DataLoader to iterate over the concatenated dataset
    data_loader = DataLoader(concatenated_dataset, batch_size=args.batch_size, shuffle=True)

    infer_model = LinearModel(n_feature=args.dimZ, n_output=1)
    infer_model = infer_model.to(args.device)
    optimizer_infer = torch.optim.Adam(infer_model.parameters(), lr=lr)

    criterion = nn.BCEWithLogitsLoss()  # Binary Cross Entropy with Logits Loss

    # Initialize AUC metric
    auroc = BinaryAUROC().to(device)
    accuracy = Accuracy(task='binary').to(device)

    # Training loop
    num_epochs = 40
    for epoch in range(num_epochs):
        infer_model.train()
        for x, y in data_loader:
            x, y = x.to(device), y.to(device).float()  # Ensure y is of type float for BCEWithLogitsLoss
            optimizer_infer.zero_grad()

            logits_z, logits_y, x_hat, mu, logvar = vib(x, mode='with_reconstruction')  # (B, C* h* w), (B, N, 10)
            y_pred = infer_model(logits_z).squeeze()  # Get predictions and squeeze to match y shape

            x = x.view(x.size(0), -1)
            # Compute the loss
            loss = criterion(y_pred, y)
            loss.backward()
            optimizer_infer.step()

            # Update AUC metric
            auroc.update(y_pred, y.int())
            accuracy.update((torch.sigmoid(y_pred) > 0.5).int(), y.int())

        # Compute AUC for the epoch
        epoch_auc = auroc.compute()
        epoch_accuracy = accuracy.compute()
        print(
            f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item()}, AUC: {epoch_auc.item()}, Accuracy: {epoch_accuracy.item()}')

        # Reset the metric for the next epoch
        auroc.reset()
        accuracy.reset()

    return infer_model

def membership_inf_results(infer_model, vib, test_data_loader, state):

    # Initialize AUC metric
    auroc = BinaryAUROC().to(device)
    accuracy = Accuracy(task='binary').to(device)

    infer_model.eval()
    for x, y in test_data_loader:
        x, y = x.to(device), y.to(device).float()  # Ensure y is of type float for BCEWithLogitsLoss


        logits_z, logits_y, x_hat, mu, logvar = vib(x, mode='with_reconstruction')  # (B, C* h* w), (B, N, 10)
        y_pred = infer_model(logits_z).squeeze()  # Get predictions and squeeze to match y shape


        # Update AUC metric
        auroc.update(y_pred, y.int())
        accuracy.update((torch.sigmoid(y_pred) > 0.5).int(), y.int())

    # Compute AUC for the epoch
    epoch_auc = auroc.compute()
    epoch_accuracy = accuracy.compute()
    print(f'{state}, AUC: {epoch_auc.item()}, Accuracy: {epoch_accuracy.item()}')

    # Reset the metric for the next epoch
    auroc.reset()
    accuracy.reset()

    return epoch_auc.item()



seed = 0
torch.cuda.manual_seed_all(seed)
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# torch.use_deterministic_algorithms(True)

# parse args
args = args_parser()
args.gpu = 0
# args.num_users = 10
args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
args.iid = True
# args.model = 'z_linear'
args.model = 'Normal'
args.num_epochs = 1
args.dataset = 'CelebA'
args.add_noise = False
args.beta = 0.0001
args.mcr_rate = 0.001
args.mse_rate = 0.1
args.lr = 0.0001
args.unlearn_learning_rate = 0.1
args.ep_distance = 20
args.dimZ =  64 #Normalize #10 /2  # 40 # 2
args.batch_size = 16
args.erased_local_r = 0.02
args.construct_size = 0.02
# args.auxiliary_size = 0.01
args.train_type = "MULTI"
args.kld_to_org = 1
args.unlearn_bce = 0.3
args.self_sharing_rate = 0.8
args.laplace_scale = 1
args.laplace_epsilon = 10
args.num_epochs_recon = 50
args.k = 2 # number of augmentations per sample

### 1: 0.000017, 20: 0.00034, 40: 0.00067, 60: 0.001, 80: 0.00134, 100: 0.00167
# print('args.beta', args.beta, 'args.lr', args.lr)

print('\n'.join(f'{k}={v}' for k, v in vars(args).items()))

device = args.device
print("device", device)

if args.dataset == 'MNIST':
    transform = T.Compose([
        T.ToTensor()
        # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    trans_mnist = transforms.Compose([transforms.ToTensor(), ])

    # Base transforms (convert to tensor and normalize)
    base_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # Augmentation transforms:
    augmentation_transform = transforms.Compose([
        transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_set = MNIST('../data/mnist', train=True, transform=None, download=True)
    test_set = MNIST('../data/mnist', train=False, transform=base_transform, download=True)
    multi_view_dataset = MultiViewMNIST(train_set, args.k, base_transform, augmentation_transform)
    original_train_set = MNIST('../data/mnist', train=True, transform=base_transform, download=True)

elif args.dataset == 'CIFAR10':
    train_transform = T.Compose([  # T.RandomCrop(32, padding=4),
        T.ToTensor(),
    ])  # T.Normalize((0.4914, 0.4822, 0.4465), (0.2464, 0.2428, 0.2608)),                                 T.RandomHorizontalFlip(),
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
    test_transform = T.Compose([T.ToTensor(), ])  # T.Normalize((0.4914, 0.4822, 0.4465), (0.2464, 0.2428, 0.2608))
    train_set = CIFAR10('../data/cifar', train=True, transform=None, download=False)
    test_set = CIFAR10('../data/cifar', train=False, transform=base_transform, download=False)
    multi_view_dataset = MultiViewCIFAR(train_set, args.k, base_transform, aug_transform)
    original_train_set = CIFAR10('../data/cifar', train=True, transform=base_transform, download=False)

elif args.dataset == 'CelebA':
    train_transform = T.Compose([T.Resize((32, 32)),
                                 T.ToTensor(),
                                 ])  # T.Normalize((0.4914, 0.4822, 0.4465), (0.2464, 0.2428, 0.2608)),                                 T.RandomHorizontalFlip(),
    test_transform = T.Compose([
        T.Resize((32, 32)),
        T.ToTensor(),
        T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])  # T.Normalize((0.4914, 0.4822, 0.4465), (0.2464, 0.2428, 0.2608))
    base_transform = T.Compose([
        T.Resize((32, 32)),
        T.ToTensor(),
        T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    aug_transform = T.Compose([
        T.Resize((32, 32)),
        # If you’d like random crop with padding:
        # T.RandomCrop(32, padding=4),
        # T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    #/kaggle/input/celeba/
    data_path = '../data/CelebA'
    train_set = CelebA(data_path, split='train', target_type = 'attr', transform=None, download=False)
    test_set = CelebA(data_path, split='test', target_type = 'attr', transform=base_transform, download=False)
    multi_view_dataset = MultiViewCelebA(train_set, args.k, base_transform, aug_transform)
    original_train_set = CelebA(data_path, split='train', transform=base_transform, download=False)





train_loader = DataLoader(multi_view_dataset, batch_size=args.batch_size, shuffle=True)
original_train_loader = DataLoader(original_train_set, batch_size=args.batch_size, shuffle=False)
test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=True )

full_size = len(multi_view_dataset)
original_size = len(original_train_set)
print("full size", full_size, "original_size", original_size)



vib, lr = init_vib(args)
vib.to(args.device)

loss_fn = nn.CrossEntropyLoss()

reconstruction_function = nn.MSELoss(size_average=True)

acc_test = []
print("learning")

print('Training VIBI')
print(f'{type(vib.encoder).__name__:>10} encoder params:\t{num_params(vib.encoder) / 1000:.2f} K')
print(f'{type(vib.approximator).__name__:>10} approximator params:\t{num_params(vib.approximator) / 1000:.2f} K')
print(f'{type(vib.decoder).__name__:>10} decoder params:\t{num_params(vib.decoder) / 1000:.2f} K')
# inspect_explanations()

# train VIB
clean_acc_list = []
mse_list = []

train_type = args.train_type

# we first train the model with the erasing data, and then we will unlearn it
start_time = time.time()
for epoch in range(args.num_epochs):
    vib.train()
    vib = vib_train(train_loader, vib, loss_fn, reconstruction_function, args, epoch, train_type)  # dataloader_total, dataloader_w_o_twin


print('acc list', clean_acc_list)
print('mse list', mse_list)
end_time = time.time()
running_time = end_time - start_time
print(f'VIB Training took {running_time} seconds')

# train infer model

infer_model = get_membership_inf_model(original_train_set, test_set, vib, args)

########

vib.eval()
acc_t = eva_vib(vib, test_loader, args, name='on test dataset after unlearning', epoch=999)
acc_r = eva_vib(vib, original_train_loader, args, name='on the remaining training after unlearning', epoch=999)



vib.eval()
all_embeddings = []
all_labels = []

with torch.no_grad():
    for step, (images, labels) in enumerate(original_train_loader):
        images = images.to(device)
        labels = labels[:,20].to(device)
        z, logits_y, x_hat, mu, logvar = vib(images, mode='with_reconstruction')  # (B, C* h* w), (B, N, 10)
        z = F.normalize(z, p=2, dim=1)
        all_embeddings.append(z.cuda())
        all_labels.append(labels.cuda())
        # print(z.shape)

# Combine embeddings and labels into single tensors
all_embeddings = torch.cat(all_embeddings, dim=0)  # [N, d]
all_labels = torch.cat(all_labels, dim=0)          # [N]

# Filter embeddings for label = 1
label_1_mask = (all_labels == 1)
label_1_embeddings = all_embeddings[label_1_mask]

# Compute centroid (mean of embeddings for label = 1)
if label_1_embeddings.size(0) > 0:
    centroid_z = label_1_embeddings.mean(dim=0)  # Shape: [embedding_dim]
    print(f"Centroid for label = 1: {centroid_z}")
else:
    print("No samples with label = 1 found in the dataset.")

unlearning_samples = 10

unlearn_indices = torch.where(label_1_mask)[0][:unlearning_samples]  # Example: Select the first 10 samples

# Ensure unlearn_embeddings are collected for the specified indices
unlearn_embeddings = all_embeddings[unlearn_indices]

# Calculate distances between the centroid and each embedding in unlearn_indices
distances = pairwise_distance(unlearn_embeddings, centroid_z.unsqueeze(0))  # Shape: [len(unlearn_indices)]

print("distance before unlearning: ",distances.mean())


unlearn_images = [original_train_loader.dataset[i][0].to(device) for i in unlearn_indices]
unlearn_images_with_label = [original_train_loader.dataset[i] for i in unlearn_indices]

unlearning_data_loader = DataLoader(unlearn_images_with_label, batch_size=args.batch_size, shuffle=True)

labeled_unlearn_set = CustomLabelDataset(unlearn_images_with_label, 0) # we set label to 0 as the unlearned should not in the training set

concatenated_dataset = ConcatDataset([labeled_unlearn_set, labeled_unlearn_set])
unl_infer_data_loader = DataLoader(concatenated_dataset, batch_size=args.batch_size, shuffle=True)

# calculate berfore unlearning
infer_acc = membership_inf_results(infer_model, vib, unl_infer_data_loader, "Before unl")



# Convert unlearn_indices to NumPy for indexing the t-SNE result
unlearn_indices_np = unlearn_indices.cpu().numpy()

# Limit the number of samples for t-SNE (optional)
N = 5000
all_embeddings2 = all_embeddings[:N]
all_labels2 = all_labels[:N]

# Dimensionality Reduction with t-SNE (still using sklearn)
tsne = TSNE(n_components=2, learning_rate='auto', init='pca', random_state=42)
emb_2d = tsne.fit_transform(all_embeddings2.cpu().numpy())  # sklearn requires NumPy

# Plot the 2D embeddings, here we show the distribution before unlearning,
plt.figure(figsize=(10, 10))
scatter = plt.scatter(emb_2d[:, 0], emb_2d[:, 1], c=all_labels2.cpu().numpy(), cmap='tab10', alpha=0.7, s=10)

# Highlight the selected samples (unlearn_indices)
plt.scatter(
    emb_2d[unlearn_indices_np, 0],  # x-coordinates of selected samples
    emb_2d[unlearn_indices_np, 1],  # y-coordinates of selected samples
    c='red',  # Color for highlighted samples
    marker='x',  # Marker style for highlighted samples
    s=150,  # Marker size
    label="Selected Samples"
)

plt.colorbar(scatter, ticks=range(10))
plt.title("2D Visualization of Learned Representations (MMCR)")
plt.xlabel("Dimension 1")
plt.ylabel("Dimension 2")

plt.savefig("sine_wave.png", dpi=300)  # Save with high resolution
# plt.show()



# 为对比损失做准备
# Positives: 同类但未被unlearn_indices选中的
remaining_label_1_indices = torch.where(label_1_mask)[0]
# 去掉unlearn的indices，剩下的作为potential positives
positives_indices = remaining_label_1_indices[~torch.isin(remaining_label_1_indices, unlearn_indices)]
positives_embeddings = all_embeddings[positives_indices].to(device)

# Negatives: 全部erased类的数据 (label=1类)可作为负样本
negatives_indices = torch.where(label_1_mask)[0]
negatives_embeddings = all_embeddings[negatives_indices].to(device)

tau = 0.07  # 温度参数，可调

optimizer = torch.optim.Adam(vib.parameters(), lr=args.lr)


# record time for unlearning
start_time = time.time()

epochs = 2
for epoch in range(epochs):
    vib.train()  # Switch to training mode

    for images, labels in unlearning_data_loader:
        images, labels = images.to(device), labels[:,20].to(device)

        z_unlearn, logits_y, x_hat, mu, logvar = vib(images, mode='with_reconstruction')


        KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar).cuda()
        KLD_mean = torch.mean(KLD_element).mul_(-0.5).cuda()

        images = images.view(images.size(0), -1)
        BCE = reconstruction_function(x_hat, images)
        z_unlearn = F.normalize(z_unlearn, p=2, dim=1)

        # Loss: Maximize the distance to the centroid
        distance_loss = torch.norm(z_unlearn - centroid_z.to(device), p=2, dim=1).mean()  # Maximize distance

        H_p_q = loss_fn(logits_y, labels)

        loss = args.beta * KLD_mean -BCE * 0.001  - H_p_q * 0.001

        print("distance", distance_loss.item() )
        # Backpropagation and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item()}")

print("Unlearning completed.")

end_time = time.time()
running_time = end_time - start_time
print(f'unlearning with dp {running_time} seconds')

# calculate after unlearning
infer_acc = membership_inf_results(infer_model, vib, unl_infer_data_loader, "after unl")

# Step 1: Compute updated embeddings after unlearning
vib.eval()
all_embeddings_after = []
all_labels_after = []
with torch.no_grad():
    for step, (images, labels) in enumerate(original_train_loader):
        images = images.to(device)
        labels = labels[:,20].to(device)
        z, _, _, _, _ = vib(images, mode='with_reconstruction')
        z = F.normalize(z, p=2, dim=1)
        all_embeddings_after.append(z.cuda())
        all_labels_after.append(labels.cuda())

all_embeddings_after = torch.cat(all_embeddings_after, dim=0)  # Combine into a single tensor


# Ensure unlearn_embeddings are collected for the specified indices
unlearn_embeddings = all_embeddings_after[unlearn_indices]

# Calculate distances between the centroid and each embedding in unlearn_indices
distances_after = pairwise_distance(unlearn_embeddings, centroid_z.unsqueeze(0))  # Shape: [len(unlearn_indices)]

print("distance after unlearning: ", distances_after.mean(), "distance dif:", distances_after.mean() - distances.mean())


# Step 2: Perform t-SNE on the updated embeddings
# tsne_after = TSNE(n_components=2, learning_rate='auto', init='pca', random_state=42)
emb_2d_after = tsne.fit_transform(all_embeddings_after[:N].cpu().numpy())  # sklearn requires NumPy
all_labels_after = torch.cat(all_labels_after, dim=0)
all_labels_2d_after = all_labels_after[:N]
# Step 3: Highlight unlearned samples
unlearn_indices_np = unlearn_indices.cpu().numpy()


plt.figure(figsize=(10, 10))
scatter = plt.scatter(
    emb_2d_after[:, 0], emb_2d_after[:, 1], c=all_labels_2d_after.cpu().numpy(), cmap='tab10', alpha=0.7, s=10, label="All Samples"
)

# Highlight the unlearned samples
plt.scatter(
    emb_2d_after[unlearn_indices_np, 0],  # x-coordinates of unlearned samples
    emb_2d_after[unlearn_indices_np, 1],  # y-coordinates of unlearned samples
    c='red',
    marker='x',
    s=150,
    label="Unlearned Samples"
)

# Add a legend and color bar
plt.colorbar(scatter, ticks=range(10))
plt.title("2D Visualization of Learned Representations (After Unlearning)")
plt.xlabel("Dimension 1")
plt.ylabel("Dimension 2")
plt.legend()



# Save and show the plot
plt.savefig("highlighted_after_unlearning.png", dpi=300)  # Save with high resolution
# plt.show()


vib.eval()
acc_t = eva_vib(vib, test_loader, args, name='on test dataset after unlearning', epoch=999)
acc_r = eva_vib(vib, original_train_loader, args, name='on the remaining training after unlearning', epoch=999)





