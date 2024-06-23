import pytorch_lightning as pl
from torchvision import transforms
import sys
import os
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

from ssl_methods.data_modules import ReconstructionDataModule
from ssl_methods.bigan.train_module import BiGAN

torch.cuda.empty_cache()
transform = transforms.Compose([
    transforms.ToTensor()
])
dm = ReconstructionDataModule("data", transform)
dm.setup()
training_module = BiGAN()
checkpoint = pl.callbacks.ModelCheckpoint(monitor="train_d_loss", mode="max", dirpath="lightning_logs", filename="bigan-{epoch}-{train_d_loss:.2f}-{train_g_loss:.2f}")
trainer = pl.Trainer(max_epochs=50, callbacks=[checkpoint])
trainer.fit(training_module, dm)