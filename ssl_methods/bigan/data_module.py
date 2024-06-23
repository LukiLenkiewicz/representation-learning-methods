from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torchvision import transforms
from torchvision.datasets import STL10

import pytorch_lightning as pl


class STLDataModule(pl.LightningDataModule):
    def __init__(self, batch_size=16):
        super().__init__()
        self.batch_size = batch_size
        self.transform = transforms.Compose(
            [transforms.Resize((32, 32)), transforms.ToTensor()]
        )
        self.path = "../data/bigan"

    def setup(self, stage=None):
        dataset = STL10(self.path, split="train", transform=self.transform)
        self.train_dataset, self.val_dataset = random_split(dataset, [0.8, 0.2])
        self.test_dataset = STL10(self.path, split="test", transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size)
