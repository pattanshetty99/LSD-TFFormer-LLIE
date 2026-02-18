import torch
import torch.nn as nn
from .blocks import ConvBlock, TransformerBlock

class IlluminationEstimator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            ConvBlock(3, 32),
            ConvBlock(32, 32),
            nn.Conv2d(32, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)


class ReflectanceRestoration(nn.Module):
    def __init__(self):
        super().__init__()
        self.head = ConvBlock(3, 64)

        self.body = nn.Sequential(
            TransformerBlock(64),
            TransformerBlock(64),
            TransformerBlock(64),
            TransformerBlock(64)
        )

        self.tail = nn.Conv2d(64, 3, 3, padding=1)

    def forward(self, x):
        x = self.head(x)
        x = self.body(x)
        return self.tail(x)


class Denoiser(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            ConvBlock(3, 64),
            ConvBlock(64, 64),
            ConvBlock(64, 64),
            nn.Conv2d(64, 3, 3, padding=1)
        )

    def forward(self, x):
        return x + self.model(x)


class LSD_TFFormer(nn.Module):
    def __init__(self):
        super().__init__()
        self.illum = IlluminationEstimator()
        self.reflectance = ReflectanceRestoration()
        self.denoiser = Denoiser()

    def forward(self, x):
        illum = self.illum(x)
        illum = torch.clamp(illum, min=0.1)

        reflectance = x / illum
        restored = self.reflectance(reflectance)
        denoised = self.denoiser(restored)

        out = torch.clamp(denoised * illum, 0, 1)

        return out, illum
