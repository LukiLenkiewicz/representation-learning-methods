import torch
import torch.nn as nn


class NLLLoss(nn.Module):
    def __init__(self, reduction="mean"):
        super().__init__()
        if reduction.lower() not in ["mean", "sum", "none"]:
            raise NotImplementedError
        self.reduction = reduction.lower()

    def forward(self, noise_out, logdets):
        loss = 0.5 * torch.sum(noise_out * noise_out, dim=1) - torch.sum(logdets, dim=1)

        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()
        elif self.reduction == "none":
            pass
        else:
            pass
        return loss
