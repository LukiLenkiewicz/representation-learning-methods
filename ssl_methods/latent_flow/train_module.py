import torch.nn as nn
import torch.optim as optim

import pytorch_lightning as pl

from ssl_methods.encoder import ResNet18Encoder
from ssl_methods.decoder import ResNet18Decoder
from ssl_methods.latent_flow.flow import Flow
from ssl_methods.train_modules import BasePretrainingModule


class LatentFlowPretrainingModule(pl.LightningModule):
    def __init__(self, learning_rate=1e-3):
        super().__init__()
        self.loss_fn = nn.MSELoss()
        self.learning_rate = learning_rate

        self.encoder = ResNet18Encoder()
        self.decoder = ResNet18Decoder()
        self.flow = Flow()

    def forward(self, img):
        z = self.encoder(img)
        y = self.decoder(z)
        return y

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
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def test_step(self, batch, batch_idx):
        img, label = batch
        output = self(img)
        loss = self.loss_fn(output, img)
        self.log('test_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        encoder_optimizer = optim.Adam(self.encoder.parameters(), lr=self.learning_rate)
        decoder_optimizer = optim.Adam(self.decoder.parameters(), lr=self.learning_rate)
        flow_optimizer = optim.Adam(self.flow.parameters(), lr=self.learning_rate)
        return encoder_optimizer, decoder_optimizer, flow_optimizer
