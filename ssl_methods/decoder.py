import torch.nn as nn
import torch.nn.functional as F


class ResNet18Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.in_channels = 512
        self.conv1 = nn.ConvTranspose2d(64, 3, kernel_size=7, stride=2, padding=3, bias=False)

        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")

        self.layer4 = self._make_layer(512, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer1 = self._make_layer(64, 2, stride=1)

    def _make_layer(self, out_channels, blocks, stride):
        upsample = None
        if stride != 1 or self.in_channels != out_channels * BasicDecoderBlock.expansion:
            upsample = nn.Sequential(
                nn.ConvTranspose2d(self.in_channels, out_channels * BasicDecoderBlock.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * BasicDecoderBlock.expansion),
            )

        layers = []
        layers.append(BasicDecoderBlock(self.in_channels, out_channels, stride, upsample))
        self.in_channels = out_channels * BasicDecoderBlock.expansion
        for _ in range(1, blocks):
            layers.append(BasicDecoderBlock(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, z):
        x = z.reshape(-1, 512, 3, 3)

        x = self.layer4(x)
        x = F.interpolate(x, size=(6, 6))
        x = self.layer3(x)
        x = F.interpolate(x, size=(12, 12))
        x = self.layer2(x)
        x = F.interpolate(x, size=(24, 24))
        x = self.layer1(x)

        x = self.relu(x)
        x = self.upsample(x)
        x = self.bn1(x)

        x = self.conv1(x)
        x = F.interpolate(x, size=(96, 96))

        return x


class BasicDecoderBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.ConvTranspose2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out