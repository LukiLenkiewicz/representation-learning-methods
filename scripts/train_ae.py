from torchvision import transforms
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from ssl_methods.data_modules import ReconstructionDataModule
from ssl_methods.train_modules import BasePretrainingModule


def main():
    preprocess = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )

    data_module = ReconstructionDataModule("./data", preprocess)
    data_module.setup()

    training_module = BasePretrainingModule()

    checkpoint_callback = ModelCheckpoint(dirpath="./models/ae", monitor="val_loss")
    trainer = pl.Trainer(max_epochs=50, callbacks=[checkpoint_callback])
    trainer.fit(training_module, data_module)


if __name__ == "__main__":
    main()
