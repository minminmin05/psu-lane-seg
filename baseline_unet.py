"""Baseline for comparison only: the plain UNet from the course prototype code.

This is the reference network handed out with the assignment (4-level UNet,
base width 32, ~7.8M parameters). It is NOT the network designed in this repo
(that is TinyLaneNet in model.py); it is kept so both can be trained with the
same pipeline and compared.

Two small changes from the handout so it can run on a 16:9 input such as
128x72 (the original only works when H and W are multiples of 16):
  - the upsampled feature map is resized to the skip's size before concatenation
  - the input is normalized with the same mean/std buffers as TinyLaneNet
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(Conv3x3 -> BN -> ReLU) x2, preserves spatial size."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class Down(nn.Module):
    """MaxPool2x2 followed by DoubleConv."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x):
        return self.conv(self.pool(x))


class Up(nn.Module):
    """ConvTranspose2x2 upsample, concat with encoder skip, then DoubleConv."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = DoubleConv(out_ch * 2, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class UNet(nn.Module):
    """4-level UNet (base_ch -> 16 * base_ch at the bottleneck)."""

    def __init__(self, in_channels=3, num_classes=1, base_ch=32,
                 mean=(0.5, 0.5, 0.5), std=(0.25, 0.25, 0.25)):
        super().__init__()
        self.register_buffer("mean", torch.tensor(mean, dtype=torch.float32).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(std, dtype=torch.float32).view(1, 3, 1, 1))

        self.in_conv = DoubleConv(in_channels, base_ch)
        self.down1 = Down(base_ch, base_ch * 2)
        self.down2 = Down(base_ch * 2, base_ch * 4)
        self.down3 = Down(base_ch * 4, base_ch * 8)
        self.down4 = Down(base_ch * 8, base_ch * 16)

        self.up1 = Up(base_ch * 16, base_ch * 8)
        self.up2 = Up(base_ch * 8, base_ch * 4)
        self.up3 = Up(base_ch * 4, base_ch * 2)
        self.up4 = Up(base_ch * 2, base_ch)

        self.out_conv = nn.Conv2d(base_ch, num_classes, kernel_size=1)

    def forward(self, x):
        x = (x - self.mean) / self.std
        x1 = self.in_conv(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        return self.out_conv(x)
