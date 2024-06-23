import torch
import torch.nn as nn
import numpy as np
import pytorch_lightning as pl
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from ssl_methods.decoder import ResNet18Decoder
from ssl_methods.encoder import ResNet18Encoder
from bigan.loss import adversarial_loss


class Discriminator(nn.Module):
    def __init__(self, latent_dim, img_shape):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim + int(np.prod(img_shape)), 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1),
        )

    def forward(self, z, img):
        img_flat = img.view(img.size(0), -1)
        d_in = torch.cat((z, img_flat), dim=1)
        validity = self.model(d_in)
        return validity

class BiGAN(pl.LightningModule):
    def __init__(self, latent_dim=64, lr=0.0005, b1=0.5, b2=0.999):
        super(BiGAN, self).__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False
        self.encoder = ResNet18Encoder()
        self.decoder = ResNet18Decoder()
        self.discriminator = Discriminator(latent_dim, (3, 96, 96))
        self.latent_dim = latent_dim

    def forward(self, x):
        z = self.encoder(x)
        img = self.decoder(z)
        return img

    def training_step(self, batch, batch_idx):
        imgs, _ = batch

        z = torch.randn(imgs.shape[0], self.latent_dim)
        z = z.type_as(imgs)

        gen_imgs = self.decoder(z)

        encoded_imgs = self.encoder(imgs)

        validity_real = self.discriminator(encoded_imgs, imgs)
        validity_fake = self.discriminator(z, gen_imgs)

        g_loss = adversarial_loss(validity_fake, torch.ones_like(validity_fake))

        real_loss = adversarial_loss(validity_real, torch.ones_like(validity_real))
        fake_loss = adversarial_loss(validity_fake, torch.zeros_like(validity_fake))
        d_loss = (real_loss + fake_loss) / 2

        self.log('d_loss', d_loss, prog_bar=True, on_step=False, on_epoch=True, logger=True)
        self.log('g_loss', g_loss, prog_bar=True, on_step=False, on_epoch=True, logger=True)
        return {"d_loss": d_loss, "g_loss": g_loss}
    
    def validation_step(self, batch, batch_idx):
        imgs, _ = batch

        z = torch.randn(imgs.shape[0], self.latent_dim)
        z = z.type_as(imgs)

        gen_imgs = self.decoder(z)

        encoded_imgs = self.encoder(imgs)
        
        validity_real = self.discriminator(encoded_imgs, imgs)
        validity_fake = self.discriminator(z, gen_imgs)
        
        g_loss = adversarial_loss(validity_fake, torch.ones_like(validity_fake))
        real_loss = adversarial_loss(validity_real, torch.ones_like(validity_real))
        fake_loss = adversarial_loss(validity_fake, torch.zeros_like(validity_fake))
        d_loss = (real_loss + fake_loss) / 2
        
        self.log('val_d_loss', d_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_g_loss', g_loss, prog_bar=True, on_step=False, on_epoch=True, logger=True)
        return {"d_loss": d_loss, "g_loss": g_loss}

    def configure_optimizers(self):
        lr = self.hparams.lr
        b1 = self.hparams.b1
        b2 = self.hparams.b2
        opt_g = torch.optim.Adam(self.decoder.parameters(), lr=lr, betas=(b1, b2))
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=lr, betas=(b1, b2))
        return [opt_g, opt_d], []