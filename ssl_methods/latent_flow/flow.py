import torch
import torch.nn as nn


class Scale(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.scale = nn.Parameter(torch.zeros(1, in_ch))

    def forward(self, x):
        out = x * torch.exp(self.scale * 3)
        return out


class AffineCoupling(nn.Module):
    # Adapted from https://github.com/rosinality/glow-pytorch/blob/master/model.py
    def __init__(self, in_ch, hidden_size):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(in_ch // 2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, in_ch),
            Scale(in_ch),
        )

    def forward(self, x):
        z1, z2 = x.chunk(2, 1)

        log_s, t = self.net(z1).chunk(2, 1)
        s = torch.sigmoid(log_s + 2)

        y1 = z1
        y2 = (z2 + t) * s
        y = torch.cat([y1, y2], dim=1)
        logdet = torch.sum(torch.log(s).view(x.shape[0], -1), 1)
        return y, logdet

    def reverse(self, y):
        y1, y2 = y.chunk(2, 1)

        log_s, t = self.net(y1).chunk(2, 1)
        s = torch.sigmoid(log_s + 2)

        z1 = y1
        z2 = y2 / s - t
        x = torch.cat([z1, z2], dim=1)
        return x


class Permutation(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.in_ch = in_ch
        self.perm = nn.Parameter(torch.randperm(in_ch), requires_grad=False)

    def forward(self, x):
        assert x.shape[1] == self.in_ch
        out = x[:, self.perm]
        return out

    def reverse(self, x):
        assert x.shape[1] == self.in_ch
        out = x[:, torch.argsort(self.perm)]
        return out


class FlowNet(nn.Module):
    def __init__(self, nz, hidden_size, nblocks=4):
        super().__init__()
        self.nblocks = nblocks

        assert nz % 4 == 0

        self.affine_layers = nn.ModuleList(
            [AffineCoupling(nz, hidden_size) for _ in range(nblocks)]
        )
        self.perms = nn.ModuleList([Permutation(nz) for _ in range(nblocks - 1)])

    def forward(self, x):
        out = x
        logdets = []
        for i in range(self.nblocks - 1):
            out, logdet = self.affine_layers[i](out)
            out = self.perms[i](out)
            logdets.append(logdet)
        out, logdet = self.affine_layers[-1](out)
        logdets.append(logdet)
        logdets = torch.stack(logdets, dim=1)
        return out, logdets

    def reverse(self, x):
        out = self.affine_layers[-1].reverse(x)
        for i in range(self.nblocks - 1):
            out = self.perms[-1 - i].reverse(out)
            out = self.affine_layers[-2 - i].reverse(out)
        return out


def test_flow():
    nz = 20
    hidden_size = 64
    nblocks = 4

    netF = FlowNet(nz, hidden_size, nblocks)

    with torch.no_grad():
        test_input = torch.randn((2, nz))
        test_output, test_logdet = netF(test_input)
        test_input2 = netF.reverse(test_output)
        assert torch.allclose(test_input, test_input2), "Flow model is incorrect"
        del test_input, test_output, test_input2, test_logdet
