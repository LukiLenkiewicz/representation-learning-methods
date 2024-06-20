import torch
import torch.nn as nn
from torchvision import transforms
import pytorch_lightning as pl

from ssl_methods.encoder import ResNet18Encoder
from ssl_methods.decoder import ResNet18Decoder
from ssl_methods.data_modules import ReconstructionDataModule
from ssl_methods.train_modules import BasePretrainingModule


def main():
    preprocess = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )

    model = nn.Sequential(
        ResNet18Encoder(),
        ResNet18Decoder())

    data_module = ReconstructionDataModule(preprocess, "../data")
    data_module.setup()
    training_module = BasePretrainingModule(model)

    trainer = pl.Trainer(max_epochs=50)
    trainer.fit(training_module, data_module)


if __name__ == "__main__":
    main()
