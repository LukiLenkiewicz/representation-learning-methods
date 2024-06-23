import torch
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torchvision.datasets import STL10

import pytorch_lightning as pl


class LinearEvaluationDataModule(pl.LightningDataModule):
    def __init__(self, path, preprocess, batch_size=32):
        super().__init__()
        self.path = path
        self.batch_size = batch_size
        self.preprocess = preprocess

    def setup(self, stage=None):
        dataset = STL10(self.path, split="train", transform=self.preprocess)
        self.train_dataset, self.val_dataset = random_split(dataset, [.8, .2])
        self.test_dataset = STL10(self.path, split="test", transform=self.preprocess)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size)

      
class ReconstructionDataModule(pl.LightningDataModule):
    def __init__(self, path, preprocess, batch_size=32):
        super().__init__()
        self.path = path
        self.batch_size = batch_size
        self.preprocess = preprocess

    def setup(self, stage=None):
        generator = torch.Generator().manual_seed(42)
        self.dataset = STL10(self.path, split="unlabeled", transform=self.preprocess)
        self.train_dataset, self.val_dataset = random_split(self.dataset, [0.8, 0.2], generator=generator)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)
