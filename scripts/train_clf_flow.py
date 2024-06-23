import argparse
import torch.nn as nn
from torchvision import transforms
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from ssl_methods.utils import freeze_encoder
from ssl_methods.data_modules import LinearEvaluationDataModule
from ssl_methods.train_modules import TrainingModule
from ssl_methods.latent_flow.train_module import LatentFlowPretrainingModule


parser = argparse.ArgumentParser(
    description="Process the paths for an encoder and classifier."
)
parser.add_argument("--encoder_path", type=str, help="Path to the encoder file")
parser.add_argument("--classifier_path", type=str, help="Path to the classifier file")
args = parser.parse_args()


def main():
    preprocess = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )

    flow_module = LatentFlowPretrainingModule.load_from_checkpoint(
        f"./models/{args.encoder_path}"
    )
    encoder = flow_module.encoder

    encoder = freeze_encoder(encoder)

    classifier = nn.Linear(64, 10)
    model = nn.Sequential(encoder, classifier)

    data_module = LinearEvaluationDataModule("./data", preprocess)
    data_module.setup()

    training_module = TrainingModule(model)

    checkpoint_callback = ModelCheckpoint(
        dirpath=f"./models/{args.classifier_path}", monitor="val_acc", mode="max"
    )
    trainer = pl.Trainer(max_epochs=200, callbacks=[checkpoint_callback])
    trainer.fit(training_module, data_module)


if __name__ == "__main__":
    main()
