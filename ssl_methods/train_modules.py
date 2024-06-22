import torch.nn as nn
import torch.optim as optim
from torchmetrics import Accuracy

import pytorch_lightning as pl


class BasePretrainingModule(pl.LightningModule):
    def __init__(self, model, learning_rate=1e-3):
        super().__init__()
        self.model = model
        self.loss_fn = nn.MSELoss()
        self.learning_rate = learning_rate

    def forward(self, text):
        return self.model(text)

    def training_step(self, batch, batch_idx):
        img, label = batch
        output = self(img)
        loss = self.loss_fn(output, img)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        img, label = batch
        output = self(img)
        loss = self.loss_fn(output, img)
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def test_step(self, batch, batch_idx):
        img, label = batch
        output = self(img)
        loss = self.loss_fn(output, img)
        self.log('test_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer


class TrainingModule(pl.LightningModule):
    def __init__(self, model, learning_rate=1e-3):
        super().__init__()
        self.model = model
        self.loss_fn = nn.CrossEntropyLoss()
        self.learning_rate = learning_rate
        self.accuracy = Accuracy(task="multiclass", num_classes=10)

    def forward(self, text):
        return self.model(text)

    def training_step(self, batch, batch_idx):
        img, label = batch
        output = self(img)
        
        loss = self.loss_fn(output, label)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        acc = self.accuracy(output, label)
        self.log('train_acc', acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        img, label = batch
        output = self(img)
        loss = self.loss_fn(output, label)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        acc = self.accuracy(output, label)
        self.log('val_acc', acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def test_step(self, batch, batch_idx):
        img, label = batch
        output = self(img)
        loss = self.loss_fn(output, label)
        self.log('test_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        acc = self.accuracy(output, label)
        self.log('test_acc', acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer
