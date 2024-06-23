from torchvision import transforms, models
from torchvision.datasets import STL10
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import sys
import os

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
)

from ssl_methods.utils import freeze_encoder


class ResNetEncoder(nn.Module):
    def __init__(self):
        super(ResNetEncoder, self).__init__()
        resnet = models.resnet18(pretrained=True)
        self.resnet = nn.Sequential(*list(resnet.children())[:-1])
        self.fc = nn.Linear(resnet.fc.in_features, 100)

    def forward(self, x):
        with torch.no_grad():
            x = self.resnet(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.fc = nn.Linear(100, 256 * 4 * 4)
        self.deconv1 = nn.ConvTranspose2d(256, 128, 4, 2, 1)
        self.deconv2 = nn.ConvTranspose2d(128, 64, 4, 2, 1)
        self.deconv3 = nn.ConvTranspose2d(64, 3, 4, 2, 1)

    def forward(self, z):
        z = F.relu(self.fc(z))
        z = z.view(z.size(0), 256, 4, 4)
        z = F.relu(self.deconv1(z))
        z = F.relu(self.deconv2(z))
        z = torch.tanh(self.deconv3(z))
        return z


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)
        self.fc1 = nn.Linear(256 * 4 * 4, 512)
        self.fc2 = nn.Linear(512 + 100, 1)

    def forward(self, x, z):
        x = F.leaky_relu(self.conv1(x), 0.2)
        x = F.leaky_relu(self.conv2(x), 0.2)
        x = F.leaky_relu(self.conv3(x), 0.2)
        x = x.view(x.size(0), -1)
        x = F.leaky_relu(self.fc1(x), 0.2)
        x = torch.cat([x, z], dim=1)
        x = torch.sigmoid(self.fc2(x))
        return x


class BiGAN(pl.LightningModule):
    def __init__(self):
        super(BiGAN, self).__init__()
        self.encoder = ResNetEncoder()
        self.generator = Generator()
        self.discriminator = Discriminator()
        self.criterion = nn.BCELoss()
        self.automatic_optimization = False

    def forward(self, z):
        return self.generator(z)

    def configure_optimizers(self):
        lr = 0.0002
        beta1 = 0.5
        optimizerE = optim.Adam(self.encoder.parameters(), lr=lr, betas=(beta1, 0.999))
        optimizerG = optim.Adam(
            self.generator.parameters(), lr=lr, betas=(beta1, 0.999)
        )
        optimizerD = optim.Adam(
            self.discriminator.parameters(), lr=lr, betas=(beta1, 0.999)
        )
        return [optimizerE, optimizerG, optimizerD]

    def training_step(self, batch, batch_idx):
        images, _ = batch
        batch_size = images.size(0)
        real_labels = torch.ones(batch_size, 1).type_as(images)
        fake_labels = torch.zeros(batch_size, 1).type_as(images)

        optE, optG, optD = self.optimizers()

        z = torch.randn(batch_size, 100).type_as(images)
        fake_images = self.generator(z)
        real_outputs = self.discriminator(images, self.encoder(images))
        fake_outputs = self.discriminator(fake_images, z)
        d_loss_real = self.criterion(real_outputs, real_labels)
        d_loss_fake = self.criterion(fake_outputs, fake_labels)
        d_loss = d_loss_real + d_loss_fake

        optD.zero_grad()
        self.manual_backward(d_loss, retain_graph=True)
        optD.step()

        fake_outputs = self.discriminator(fake_images, z)
        g_loss = self.criterion(fake_outputs, real_labels)

        optG.zero_grad()
        optE.zero_grad()
        self.manual_backward(g_loss)
        optG.step()
        optE.step()

        self.log("d_loss", d_loss, prog_bar=True)
        self.log("g_loss", g_loss, prog_bar=True)

        return {"d_loss": d_loss, "g_loss": g_loss}


class STLDataModule(pl.LightningDataModule):
    def __init__(self, batch_size=32):
        super().__init__()
        self.batch_size = batch_size
        self.transform = transforms.Compose(
            [transforms.Resize((32, 32)), transforms.ToTensor()]
        )

    def setup(self, stage=None):
        self.stl10_train = STL10(
            root="../data", split="train", download=True, transform=self.transform
        )
        self.stl10_test = STL10(
            root="../data", split="test", download=True, transform=self.transform
        )
        self.stl10_unlabeled = STL10(
            root="../data", split="unlabeled", download=True, transform=self.transform
        )

    def train_dataloader(self):
        return DataLoader(self.stl10_train, batch_size=self.batch_size, shuffle=True)

    def test_dataloader(self):
        return DataLoader(self.stl10_test, batch_size=self.batch_size, shuffle=False)

    def unlabeled_dataloader(self):
        return DataLoader(
            self.stl10_unlabeled, batch_size=self.batch_size, shuffle=False
        )


stl10_dm = STLDataModule(batch_size=32)
stl10_dm.setup()

# bigan_model = BiGAN()
checkpoint_callback = ModelCheckpoint(
    monitor="d_loss",
    dirpath="./checkpoints-pretrained-bigan",
    filename="bigan-{epoch:02d}-{d_loss:.2f}",
    save_top_k=3,
    mode="min",
)

trainer = pl.Trainer(max_epochs=100, callbacks=[checkpoint_callback])

bigan_model = BiGAN.load_from_checkpoint(
    "../lightning_logs/checkpoints-pretrained-bigan/bigan-epoch=71-d_loss=0.00.ckpt"
)
freeze_encoder(bigan_model.encoder)
trainer.fit(bigan_model, stl10_dm)
