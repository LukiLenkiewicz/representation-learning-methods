import torch.nn as nn
import torch.optim as optim

import pytorch_lightning as pl

from ssl_methods.encoder import ResNet18Encoder
from ssl_methods.decoder import ResNet18Decoder
from ssl_methods.latent_flow.flow import FlowNet
from ssl_methods.latent_flow.loss import NLLLoss


class LatentFlowPretrainingModule(pl.LightningModule):
    def __init__(self, learning_rate=1e-3):
        super().__init__()
        self.ae_loss_fn = nn.MSELoss()
        self.flow_loss_fn = NLLLoss(reduction="mean")

        self.learning_rate = learning_rate

        self.encoder = ResNet18Encoder()
        self.decoder = ResNet18Decoder()
        self.flow = FlowNet(nz=64, hidden_size=512, nblocks=8)

    def forward(self, img):
        z = self.encoder(img)
        return z

    def training_step(self, batch, batch_idx):
        img, label = batch

        z = self(img)
        y = self.decoder(z)
        ae_loss = self.ae_loss_fn(y, img)
        noise_out, logdets = self.flow(z)
        flow_loss = self.flow_loss_fn(noise_out, logdets)

        loss = ae_loss + flow_loss
        self.log(
            "train_ae_loss",
            ae_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.log(
            "train_flow_loss",
            flow_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.log(
            "train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        return loss

    def validation_step(self, batch, batch_idx):
        img, label = batch

        z = self(img)
        y = self.decoder(z)
        ae_loss = self.ae_loss_fn(y, img)
        noise_out, logdets = self.flow(z)
        flow_loss = self.flow_loss_fn(noise_out, logdets)

        loss = ae_loss + flow_loss
        self.log(
            "val_ae_loss",
            ae_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.log(
            "val_flow_loss",
            flow_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.log(
            "val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True
        )
        return loss

    def test_step(self, batch, batch_idx):
        img, label = batch

        z = self(img)
        y = self.decoder(z)
        ae_loss = self.ae_loss_fn(y, img)
        noise_out, logdets = self.flow(z)
        flow_loss = self.flow_loss_fn(noise_out, logdets)

        loss = ae_loss + flow_loss
        self.log(
            "test_ae_loss",
            ae_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.log(
            "test_flow_loss",
            flow_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.log(
            "test_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True
        )
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer
