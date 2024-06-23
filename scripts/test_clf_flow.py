import argparse
import torch.nn as nn
from torchvision import transforms
import pytorch_lightning as pl

from ssl_methods.encoder import ResNet18Encoder
from ssl_methods.data_modules import LinearEvaluationDataModule
from ssl_methods.train_modules import TrainingModule


parser = argparse.ArgumentParser(description="Process model path.")
parser.add_argument(
    "--module_path",
    type=str,
    help="Path to trained model",
    default="./models/latent-flow-large-linear-2/latent-flow-large-2.ckpt",
)
args = parser.parse_args()


def main():
    preprocess = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )

    classifier = nn.Linear(64, 10)
    model = nn.Sequential(ResNet18Encoder(), classifier)

    data_module = LinearEvaluationDataModule("./data", preprocess)
    trainer = pl.Trainer(max_epochs=50)

    training_module = TrainingModule.load_from_checkpoint(args.module_path, model=model)
    trainer.test(training_module, data_module)


if __name__ == "__main__":
    main()
