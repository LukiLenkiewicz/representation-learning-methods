from torchvision import transforms
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

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

    checkpoint_callback = ModelCheckpoint(dirpath="./models/latent-flow-large", monitor="val_loss")
    trainer = pl.Trainer(max_epochs=50, callbacks=[checkpoint_callback])
    trainer.fit(training_module, data_module)


if __name__ == "__main__":
    main()
