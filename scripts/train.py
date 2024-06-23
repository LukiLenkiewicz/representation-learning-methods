import torch.nn as nn
import pytorch_lightning as pl

from ssl_methods.encoder import ResNet18Encoder
from ssl_methods.decoder import ResNet18Decoder
from ssl_methods.data_modules import ReconstructionDataModule
from ssl_methods.train_modules import TrainingModule


def main():
    encoder = ResNet18Encoder()
    decoder = ResNet18Decoder()
    model = nn.Sequential(encoder, decoder)

    data_module = ReconstructionDataModule("../")
    training_module = TrainingModule(model)

    trainer = pl.Trainer()
    trainer.fit(training_module, data_module)


if __name__ == "__main__":
    main()
