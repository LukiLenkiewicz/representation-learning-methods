import torch.nn as nn
import torch.optim as optim
from torchmetrics import Accuracy, F1Score
import pytorch_lightning as pl

from ssl_methods.encoder import ResNet18Encoder
from ssl_methods.decoder import ResNet18Decoder


class BasePretrainingModule(pl.LightningModule):
    def __init__(self, learning_rate=1e-3):
        super().__init__()
        self.encoder = ResNet18Encoder()
        self.decoder = ResNet18Decoder()
        self.model = nn.Sequential(self.encoder, self.decoder)
        self.loss_fn = nn.MSELoss()
        self.learning_rate = learning_rate

    def forward(self, img):
        return self.model(img)

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
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        return optimizer


class TrainingModule(pl.LightningModule):
    def __init__(self, model, learning_rate=1e-3):
        super().__init__()
        self.model = model
        self.loss_fn = nn.CrossEntropyLoss()
        self.learning_rate = learning_rate
        self.accuracy = Accuracy(task="multiclass", num_classes=10)
        self.f1 = F1Score(task="multiclass", num_classes=10, average="macro")
    def forward(self, img):
        return self.model(img)

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
        f1 = self.f1(output, label)
        self.log('test_f1', f1, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer
