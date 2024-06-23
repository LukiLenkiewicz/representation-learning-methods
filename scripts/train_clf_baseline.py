import argparse

import torch.nn as nn
from torchvision import transforms
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from ssl_methods.utils import freeze_encoder
from ssl_methods.encoder import ResNet18Encoder
from ssl_methods.data_modules import LinearEvaluationDataModule
from ssl_methods.train_modules import BasePretrainingModule, TrainingModule


parser = argparse.ArgumentParser(
    description="Process the paths for an encoder and classifier."
)
parser.add_argument(
    "--encoder_path", default=None, type=str, help="Path to the encoder file"
)
parser.add_argument("--classifier_path", type=str, help="Path to the classifier file")
parser.add_argument("--freeze", action="store_true", help="freeze encoder or not")
args = parser.parse_args()


def main():
    preprocess = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )

    if args.encoder_path:
        module = BasePretrainingModule.load_from_checkpoint(args.encoder_path)
        encoder = module.encoder
    else:
        encoder = ResNet18Encoder()

    encoder = freeze_encoder(encoder)

    classifier = nn.Linear(64, 10)

    model = nn.Sequential(encoder, classifier)

    data_module = LinearEvaluationDataModule("./data", preprocess)
    data_module.setup()

    training_module = TrainingModule(model)

    checkpoint_callback = ModelCheckpoint(
        dirpath="./models/baseline", monitor="val_acc", mode="max"
    )
    trainer = pl.Trainer(max_epochs=200, callbacks=[checkpoint_callback])
    trainer.fit(training_module, data_module)


if __name__ == "__main__":
    main()
