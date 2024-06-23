import torch.nn as nn
from torchmetrics import Accuracy, F1Score
import torch.optim as optim
import pytorch_lightning as pl

from ssl_methods.utils import freeze_encoder
from ssl_methods.bigan.train_module import BiGAN


class BiGANLinearEval(pl.LightningModule):
    def __init__(self, learning_rate=1e-3):
        super().__init__()
        self.encoder = BiGAN.load_from_checkpoint("../scripts/lightning_logs/checkpoints-pretrained-bigan/bigan-epoch=71-d_loss=0.00.ckpt").encoder
        self.model = nn.Sequential(self.encoder, nn.Linear(100, 10))
        freeze_encoder(self.encoder)
        self.loss_fn = nn.CrossEntropyLoss()
        self.learning_rate = learning_rate
        self.accuracy = Accuracy(task="multiclass", num_classes=10)
        self.f1 = F1Score(num_classes=10, task="multiclass", average="macro")

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        lr = 0.0001
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        return optimizer

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