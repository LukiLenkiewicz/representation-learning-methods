import torch.nn as nn
from torchvision import transforms
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from ssl_methods.utils import freeze_encoder
from ssl_methods.data_modules import LinearEvaluationDataModule
from ssl_methods.train_modules import TrainingModule
from ssl_methods.latent_flow.train_module import LatentFlowPretrainingModule

# TODO: fix paths

def main():
    preprocess = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )

    training_module = LatentFlowPretrainingModule.load_from_checkpoint("./models/epoch=38-step=97500.ckpt")
    encoder = freeze_encoder(training_module.encoder)
    classifier = nn.Linear(64, 10)
    model = nn.Sequential(encoder, classifier)

    data_module = LinearEvaluationDataModule("./data", preprocess)
    trainer = pl.Trainer(max_epochs=50)
    
    training_module = TrainingModule.load_from_checkpoint("./models/safe/epoch=0-step=125.ckpt", model=model)
    trainer.test(training_module, data_module)


if __name__ == "__main__":
    main()

