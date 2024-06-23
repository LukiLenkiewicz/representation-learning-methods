import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset, random_split
from torchvision.datasets import STL10, MNIST
import torchvision.transforms as transforms
import torchvision.models as models
import matplotlib.pyplot as plt
from typing import List, TypeVar
from tqdm import tqdm  # for progress bar
import numpy as np

from ssl_methods.beta_vae.components import ResNet18Decoder, ResNet18Encoder

Tensor = TypeVar("torch.tensor")


class BetaVae(nn.Module):
    num_iter = 0

    def __init__(
        self, in_channels: int, latent_dim: int, beta: int = 2, **kwargs
    ) -> None:
        super(BetaVae, self).__init__()

        self.latent_dim = latent_dim
        self.beta = beta

        self.encoder = ResNet18Encoder(latent_dim)
        self.decoder = ResNet18Decoder(latent_dim)

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input: Tensor, **kwargs) -> Tensor:
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        return [self.decode(z), input, mu, log_var]

    def encode(self, input: Tensor) -> List[Tensor]:
        return self.encoder(input)

    def decode(self, z: Tensor) -> Tensor:
        return self.decoder(z)

    def loss_function(self, *args, **kwargs) -> dict:
        recons = args[0]
        input = args[1]
        mu = args[2]
        log_var = args[3]

        # Use Gaussian negative log-likelihood for reconstruction loss
        # recons_dist = torch.distributions.Normal(input, 1.0)
        # recons_loss = recons_dist.log_prob(recons).sum()

        # recons_loss = F.mse_loss(recons, input)

        # recons = recons.view(-1, 3 * 96 * 96)
        # input = input.view(-1, 3 * 96 * 96)

        std = torch.ones_like(recons).to(recons.device)
        recons_dist = torch.distributions.Normal(recons, std)

        # Compute the log likelihood
        # recons_loss = -recons_dist.log_prob(input).mean(dim=(1, 2, 3)).mean()
        recons_loss = F.mse_loss(recons, input)

        klds = -0.5 * (1 + log_var - mu.pow(2) - log_var.exp())
        mean_kld = klds.mean(1).mean(0, True)
        loss = recons_loss + self.beta * mean_kld
        # print(loss,"STRATA")
        # print(recons_loss,"recons")
        # print(mean_kld,"kld")
        return {"loss": loss, "Reconstruction_Loss": recons_loss, "KLD": mean_kld}

    def sample(self, num_samples: int, current_device: int, **kwargs) -> Tensor:
        z = torch.randn(num_samples, self.latent_dim)
        z = z.to(current_device)
        samples = self.decode(z)
        return samples

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        return self.forward(x)[0]
