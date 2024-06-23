import torch.nn as nn
from torchvision import transforms
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from ssl_methods.utils import freeze_encoder
from ssl_methods.encoder import ResNet18Encoder
from ssl_methods.data_modules import LinearEvaluationDataModule
from ssl_methods.train_modules import TrainingModule


def main():
    preprocess = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )

    encoder = ResNet18Encoder()
    classifier = nn.Linear(64, 10)
    model = nn.Sequential(encoder, classifier)

    data_module = LinearEvaluationDataModule("./data", preprocess)
    training_module = TrainingModule(model)

    checkpoint_callback = ModelCheckpoint(dirpath="./models/baseline", monitor="val_acc", mode="max")
    trainer = pl.Trainer(max_epochs=50, callbacks=[checkpoint_callback])
    trainer.fit(training_module, data_module)
    

if __name__ == "__main__":
    main()
