import torch
from torch import nn
from torch.nn import functional as F
import torchvision.models as models


class ResNet18Encoder(nn.Module):
    def __init__(self, latent_dim):
        super(ResNet18Encoder, self).__init__()
        resnet = models.resnet18(pretrained=True)
        modules = list(resnet.children())[:-1] 
        self.resnet = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(resnet.fc.in_features, latent_dim)
        self.fc_var = nn.Linear(resnet.fc.in_features, latent_dim)

    def forward(self, x):
        x = self.resnet(x)
        x = torch.flatten(x, start_dim=1)
        mu = self.fc_mu(x)
        log_var = self.fc_var(x)
        return mu, log_var

class ResNet18Decoder(nn.Module):
    def __init__(self, latent_dim):
        super(ResNet18Decoder, self).__init__()
        self.decoder_input = nn.Linear(latent_dim, 512 * 3 * 3)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(32, 3, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.decoder_input(x)
        x = x.view(-1, 512, 3, 3)
        x = self.decoder(x)
        return x
