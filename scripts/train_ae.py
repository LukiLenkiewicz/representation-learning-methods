import torch
import torch.nn as nn
from torchvision import transforms

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

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

    data_module = ReconstructionDataModule( "./data", preprocess)
    data_module.setup()
    training_module = BasePretrainingModule(model)

    checkpoint_callback = ModelCheckpoint(dirpath="./models", monitor="val_loss")
    trainer = pl.Trainer(max_epochs=50, callbacks=[checkpoint_callback])
    trainer.fit(training_module, data_module)


if __name__ == "__main__":
    main()
