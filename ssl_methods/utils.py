import torch.nn as nn


def freeze_encoder(encoder: nn.Module) -> nn.Module:
    for param in encoder.parameters():
        param.requires_grad = False
    return encoder
