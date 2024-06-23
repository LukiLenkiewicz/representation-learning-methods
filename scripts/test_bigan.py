import torch
import pytorch_lightning as pl

from ssl_methods.bigan.eval_module import BiGANLinearEval
from ssl_methods.bigan.data_module import STLDataModule

torch.cuda.empty_cache()
stl10_dm = STLDataModule(batch_size=16)
stl10_dm.setup()

trainer = pl.Trainer(max_epochs=25)

model = BiGANLinearEval.load_from_checkpoint("../scripts/lightning_logs/checkpoints-pretrained-bigan-linear-eval-lr-x10/bigan-epoch=20-val_acc=0.33.ckpt")
trainer.test(model, stl10_dm)