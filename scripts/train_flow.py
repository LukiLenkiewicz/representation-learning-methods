from torchvision import transforms
import pytorch_lightning as pl

from ssl_methods.data_modules import ReconstructionDataModule
from ssl_methods.latent_flow.train_module import LatentFlowPretrainingModule


def main():
    preprocess = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )

    data_module = ReconstructionDataModule("./data", preprocess)
    data_module.setup()
    training_module = LatentFlowPretrainingModule()

    trainer = pl.Trainer(max_epochs=50)
    trainer.fit(training_module, data_module)


if __name__ == "__main__":
    main()
