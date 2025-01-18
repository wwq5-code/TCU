import argparse
import os
import warnings
from pathlib import Path

import torch
from lightning import Trainer, seed_everything
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal
import torch.nn as nn
from typing import Callable

from typing_extensions import Self

import torch
import torchmetrics
from lightning import LightningModule
from torch import nn, Tensor
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

import torch
import torchvision.transforms.v2 as transforms
from PIL import Image
from lightning import LightningDataModule
from torch import Tensor
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CIFAR10


@dataclass(frozen=True)
class PretrainConfig:
    dataset: Path | str = 'cifar10'
    batch_size: int = 64
    num_views: int = 16
    max_epochs: int = 500
    warmup_duration: float = 0.1
    num_workers: int = os.cpu_count() - 2
    learning_rate: float = 1e-3
    projection_dim: int = 128
    num_neighbours: int = 200
    temperature: float = 0.5
    dev: bool = False
    compile: bool = False

    @classmethod
    def from_command_line(cls, arguments: argparse.Namespace) -> Self:
        return cls(**vars(arguments))


@dataclass(frozen=True)
class LinearEvaluateConfig:
    checkpoint: Path | str
    dataset: Path | str = 'cifar10'
    batch_size: int = 512
    max_epochs: int = 50
    num_workers: int = os.cpu_count() - 2
    warmup_duration: float = 0
    learning_rate: float = 1e-3
    compile: bool = False

    @classmethod
    def from_command_line(cls, arguments: argparse.Namespace) -> Self:
        return cls(**vars(arguments))


BASIC = transforms.Compose([
    transforms.ToImage(),
    transforms.ToDtype(torch.float32, scale=True),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

AUGMENTATION = transforms.Compose([
    transforms.RandomResizedCrop(32),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
    transforms.RandomGrayscale(),
    transforms.RandomApply([transforms.GaussianBlur(3)], p=0.1),
    transforms.RandomSolarize(0.5, p=0.2),
    BASIC
])


class MultiviewDataset(CIFAR10):

    def __init__(self, root: str, train: bool = True, transform: Callable = None, num_views: int = 1) -> None:
        super().__init__(root, train, transform, download=True)
        assert num_views >= 1, 'Number of views must be larger than zero'
        self.num_views = num_views

    def __getitem__(self, index: int) -> Tensor:
        image = self.data[index]
        image = Image.fromarray(image)

        views = [image] * self.num_views

        if self.transform:
            views = [self.transform(view) for view in views]
            views = torch.stack(views)

        return views


class PretrainDataModule(LightningDataModule):

    def __init__(self, config: PretrainConfig) -> None:
        super().__init__()

        self.train_batch_size = config.batch_size
        self.valid_batch_size = config.batch_size * config.num_views
        self.num_workers = config.num_workers

        root = config.dataset

        self.train_dataset = MultiviewDataset(root, transform=AUGMENTATION, train=True, num_views=config.num_views)
        self.valid_source_dataset = CIFAR10(root, transform=BASIC, download=True, train=True)
        self.valid_target_dataset = CIFAR10(root, transform=BASIC, download=True, train=False)

        if config.dev:
            train_size = len(self.train_dataset) // 10
            valid_size = len(self.valid_target_dataset) // 10

            self.train_dataset = Subset(self.train_dataset, range(train_size))
            self.valid_source_dataset = Subset(self.valid_source_dataset, range(train_size))
            self.valid_target_dataset = Subset(self.valid_target_dataset, range(valid_size))

            print(f'[Dataset] Using dev version: train size: {train_size}, valid size: {valid_size}')

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_dataset, self.train_batch_size, num_workers=self.num_workers, shuffle=True)

    def val_dataloader(self) -> list[DataLoader]:
        return [
            DataLoader(self.valid_source_dataset, batch_size=self.valid_batch_size, num_workers=self.num_workers),
            DataLoader(self.valid_target_dataset, batch_size=self.valid_batch_size, num_workers=self.num_workers)
        ]

    def predict_dataloader(self) -> list[DataLoader]:
        return self.val_dataloader()


class LinearEvaluateDataModule(LightningDataModule):

    def __init__(self, config: LinearEvaluateConfig) -> None:
        super().__init__()

        self.batch_size = config.batch_size
        self.num_workers = config.num_workers

        self.train_dataset = CIFAR10(config.dataset, download=True, train=True, transform=AUGMENTATION)
        self.valid_dataset = CIFAR10(config.dataset, download=True, train=False, transform=BASIC)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_dataset, self.batch_size, num_workers=self.num_workers, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.valid_dataset, self.batch_size, num_workers=self.num_workers)

    def predict_dataloader(self) -> DataLoader:
        return self.val_dataloader()


class ResNet(nn.Module):

    def __init__(self, projection_dim: int):
        super().__init__()

        backbone = torchvision.models.resnet18()
        backbone.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False)
        backbone = [layer for name, layer in backbone.named_children() if name not in {'maxpool', 'fc'}]

        self.backbone = nn.Sequential(*backbone)
        self.projector = nn.Sequential(
            nn.Linear(512, 2 * projection_dim, bias=False),
            nn.BatchNorm1d(2 * projection_dim),
            nn.ReLU(),
            nn.Linear(2 * projection_dim, projection_dim)
        )

    def forward(self, inputs: Tensor) -> tuple[Tensor, Tensor]:
        embeddings = self.backbone(inputs)
        embeddings = torch.flatten(embeddings, start_dim=1)

        projections = self.projector(embeddings)

        return projections, embeddings


class ResNetForClassification(nn.Module):

    def __init__(self, num_classes: int) -> None:
        super().__init__()

        backbone = torchvision.models.resnet18()
        backbone.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False)
        backbone = [layer for name, layer in backbone.named_children() if name not in {'maxpool', 'fc'}]

        self.backbone = nn.Sequential(*backbone)
        self.classifier = nn.Linear(512, num_classes)

    @classmethod
    def from_pretrained(cls, checkpoint: str, num_classes: int = 10) -> Self:
        checkpoint = torch.load(checkpoint)
        checkpoint = {name: parameter for name, parameter in checkpoint.items() if 'backbone' in name}

        model = cls(num_classes)

        missing = model.load_state_dict(checkpoint, strict=False)

        if not all('classifier' in key for key in missing.missing_keys):
            raise ValueError(f'Unexpected missing keys: {missing.missing_keys}')

        return model

    def freeze_backbone(self) -> None:
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False

    def forward(self, inputs: Tensor) -> Tensor:
        embeddings = self.backbone(inputs)
        embeddings = torch.flatten(embeddings, start_dim=1)

        return self.classifier(embeddings)


class KNearestNeighbours:

    def __init__(self, num_neighbours: int = 200, temperature: float = 0.5, num_labels: int = 10):
        self.num_neighbours = num_neighbours
        self.temperature = temperature
        self.num_labels = num_labels

        self.train_embeddings = []
        self.train_labels = []

    def add(self, embeddings: Tensor, labels: Tensor) -> None:
        self.train_embeddings.append(embeddings)
        self.train_labels.append(labels)

    def reset(self) -> None:
        self.train_embeddings = []
        self.train_labels = []

    def score(self, embeddings: Tensor) -> Tensor:
        train_embeddings = torch.concat(self.train_embeddings)
        similarity = torchmetrics.functional.pairwise_cosine_similarity(embeddings, train_embeddings)

        weights, indices = similarity.topk(k=self.num_neighbours, dim=-1)
        weights = torch.exp(weights / self.temperature)

        labels = torch.concat(self.train_labels).expand(embeddings.size(0), -1)
        labels = torch.gather(labels, dim=-1, index=indices)
        labels = torch.nn.functional.one_hot(labels, num_classes=self.num_labels)

        scores = torch.sum(labels * weights.unsqueeze(-1), dim=1)

        return scores


def maximum_manifold_capacity(embeddings: Tensor, gamma: float) -> Tensor:
    embeddings = torch.nn.functional.normalize(embeddings, dim=-1)
    embeddings = torch.transpose(embeddings, 1, 2)

    centroids = embeddings.mean(dim=-1)

    loss = -1 * torch.linalg.matrix_norm(centroids, ord='nuc')

    if gamma > 0:
        loss += gamma * torch.linalg.matrix_norm(embeddings, ord='nuc').mean()

    return loss


class PretrainModule(LightningModule):

    def __init__(self, model: nn.Module | Callable, knn: KNearestNeighbours, config: PretrainConfig) -> None:
        super().__init__()
        self.model = model
        self.knn = knn

        self.max_epochs = config.max_epochs
        self.warmup_epochs = round(config.max_epochs * config.warmup_duration)
        self.learning_rate = config.learning_rate
        self.num_views = config.num_views

        self.top1_accuracy_valid = torchmetrics.Accuracy(num_classes=10, task='multiclass', top_k=1)
        self.top5_accuracy_valid = torchmetrics.Accuracy(num_classes=10, task='multiclass', top_k=5)

    def configure_optimizers(self) -> (list[Optimizer], list[LRScheduler]):
        optimizer = torch.optim.Adam(self.model.parameters(), self.learning_rate, weight_decay=1e-6)
        scheduler = utils.cosine_with_warmup(optimizer, self.warmup_epochs, self.max_epochs)

        return [optimizer], [scheduler]

    def training_step(self, batch: Tensor) -> Tensor:
        views = torch.flatten(batch, end_dim=1)

        projections, _ = self.model(views)
        projections = projections.view(-1, self.num_views, projections.size(-1))

        loss = maximum_manifold_capacity(projections, gamma=0)

        self.log('Train|Loss', loss, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch: tuple[Tensor, Tensor], index: int, dataloader_idx: int = 0) -> None:
        images, labels = batch

        _, embeddings = self.model(images)

        if dataloader_idx == 0:
            self.knn.add(embeddings, labels)

        if dataloader_idx == 1:
            scores = self.knn.score(embeddings)

            self.top1_accuracy_valid(scores, labels)
            self.top5_accuracy_valid(scores, labels)
            self.log('Valid|Top1 Accuracy', self.top1_accuracy_valid, on_epoch=True, add_dataloader_idx=False)
            self.log('Valid|Top5 Accuracy', self.top5_accuracy_valid, on_epoch=True, add_dataloader_idx=False)

    def on_validation_end(self) -> None:
        self.knn.reset()


class LinearEvaluateModule(LightningModule):

    def __init__(self, model: nn.Module | Callable, config: LinearEvaluateConfig) -> None:
        super().__init__()
        self.model = model

        self.max_epochs = config.max_epochs
        self.warmup_epochs = round(config.max_epochs * config.warmup_duration)
        self.learning_rate = config.learning_rate

        self.top1_accuracy_valid = torchmetrics.Accuracy(num_classes=10, task='multiclass', top_k=1)
        self.top5_accuracy_valid = torchmetrics.Accuracy(num_classes=10, task='multiclass', top_k=5)

    def configure_optimizers(self) -> (list[Optimizer], list[LRScheduler]):
        optimizer = torch.optim.Adam(self.model.parameters(), self.learning_rate, weight_decay=1e-6)
        scheduler = utils.cosine_with_warmup(optimizer, self.warmup_epochs, self.max_epochs)

        return [optimizer], [scheduler]

    def training_step(self, batch: tuple[Tensor, Tensor]) -> Tensor:
        images, labels = batch

        logits = self.model(images)
        loss = torch.nn.functional.cross_entropy(logits, labels)

        self.log('Train|Loss', loss, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch: tuple[Tensor, Tensor], index: int) -> None:
        images, labels = batch

        logits = self.model(images)
        loss = torch.nn.functional.cross_entropy(logits, labels)

        scores = torch.softmax(logits, dim=1)

        self.top1_accuracy_valid(scores, labels)
        self.top5_accuracy_valid(scores, labels)

        self.log('Valid|Loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('Valid|Top1 Accuracy', self.top1_accuracy_valid, on_step=False, on_epoch=True)
        self.log('Valid|Top5 Accuracy', self.top5_accuracy_valid, on_step=False, on_epoch=True)

torch.set_float32_matmul_precision('medium')


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=Path, default='cifar10')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--num-views', type=int, default=16)
    parser.add_argument('--max-epochs', type=int, default=500)
    parser.add_argument('--num-workers', type=int, default=os.cpu_count() - 2)
    parser.add_argument('--learning-rate', type=float, default=1e-3)
    parser.add_argument('--projection-dim', type=int, default=128)
    parser.add_argument('--num-neighbours', type=int, default=200)
    parser.add_argument('--temperature', type=float, default=0.5)
    parser.add_argument('--warmup-duration', type=float, default=0.1)
    parser.add_argument('--dev', action='store_true')
    parser.add_argument('--compile', action='store_true')

    return parser.parse_args()


def silence_compilation_warnings() -> None:
    for line in [110, 111, 117, 118]:
        warnings.filterwarnings('ignore', category=UserWarning, module='torch.overrides', lineno=line)


def pretrain():
    seed_everything(42)

    config = PretrainConfig.from_command_line(parse_arguments())
    data = PretrainDataModule(config)

    knn = KNearestNeighbours(config.num_neighbours, config.temperature)
    model = ResNet(config.projection_dim)

    if config.compile:
        model = torch.compile(model)
        silence_compilation_warnings()

    model = PretrainModule(model, knn, config)

    callbacks = [
        LearningRateMonitor(logging_interval='epoch'),
        ModelCheckpoint(
            monitor='Valid|Top1 Accuracy',
            save_top_k=1,
            save_last=True,
            mode='max',
            verbose=True,
            filename='{epoch}-{Valid|Top1 Accuracy:.2f}',
        ),
    ]

    trainer = Trainer(
        accelerator='gpu',
        devices=1,
        precision='16-mixed',
        max_epochs=config.max_epochs,
        logger=TensorBoardLogger(save_dir='logs', name=''),
        callbacks=callbacks,
        deterministic=True
    )

    trainer.fit(model, datamodule=data)


if __name__ == '__main__':
    pretrain()