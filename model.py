"""TinyLaneNet: a small encoder-decoder CNN designed from scratch for single-class lane segmentation.

Input:  3 x H x W RGB image in [0, 1]   (default 3 x 72 x 128, i.e. 128x72, 16:9)
Output: 1 x H x W lane logits           (apply sigmoid for probabilities, > 0.5 for the binary mask)

No pretrained weights are used anywhere: every layer starts from random
(Kaiming) initialization and is trained on the lane dataset only.

The README explains the reasoning behind each choice.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNAct(nn.Sequential):
    """Conv -> BatchNorm -> (ReLU). Padding keeps the spatial size at stride 1."""

    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, dilation=1, groups=1, act=True):
        padding = dilation * (kernel_size - 1) // 2
        layers = [
            nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, dilation=dilation, groups=groups, bias=False),
            nn.BatchNorm2d(out_ch),
        ]
        if act:
            layers.append(nn.ReLU(inplace=True))
        super().__init__(*layers)


class DSConv(nn.Sequential):
    """Depthwise-separable conv: depthwise 3x3 (spatial) then pointwise 1x1 (channel mixing).

    Costs 9*C_in + C_in*C_out weights instead of 9*C_in*C_out for a plain 3x3 conv.
    """

    def __init__(self, in_ch, out_ch, stride=1, dilation=1, act=True):
        super().__init__(
            ConvBNAct(in_ch, in_ch, 3, stride, dilation, groups=in_ch),
            ConvBNAct(in_ch, out_ch, 1, act=act),
        )


class DSResBlock(nn.Module):
    """Residual block around one depthwise-separable conv: relu(x + DSConv(x))."""

    def __init__(self, ch, dilation=1):
        super().__init__()
        self.conv = DSConv(ch, ch, dilation=dilation, act=False)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(x + self.conv(x))


class GlobalContext(nn.Module):
    """Squeeze-and-excitation gate: re-weights the channels using the image-wide average."""

    def __init__(self, ch, reduction=4):
        super().__init__()
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(ch, ch // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch // reduction, ch, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.gate(x)


class Up(nn.Module):
    """Bilinear upsample to the skip's size, concat the skip, fuse with 1x1, refine with a DSResBlock."""

    def __init__(self, in_ch, skip_ch, out_ch, use_skip=True):
        super().__init__()
        self.use_skip = use_skip
        self.fuse = ConvBNAct(in_ch + (skip_ch if use_skip else 0), out_ch, 1)
        self.refine = DSResBlock(out_ch)

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        if self.use_skip:
            x = torch.cat([x, skip], dim=1)
        return self.refine(self.fuse(x))


class TinyLaneNet(nn.Module):
    """3-level encoder-decoder; for a 128x72 input the feature maps are 128x72 -> 64x36 -> 32x18 -> 16x9.

    `widths`, `dilations`, `global_context` and `skips` exist so the ablations in the
    README can be reproduced; the defaults are the final design.
    """

    def __init__(self, widths=(16, 32, 64, 96), depths=(1, 2), dilations=(1, 2, 3),
                 global_context=True, skips=True,
                 mean=(0.5, 0.5, 0.5), std=(0.25, 0.25, 0.25)):
        super().__init__()
        c0, c1, c2, c3 = widths
        # Input normalization is part of the model, so callers always pass RGB in [0, 1].
        self.register_buffer("mean", torch.tensor(mean, dtype=torch.float32).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(std, dtype=torch.float32).view(1, 3, 1, 1))

        self.stem = ConvBNAct(3, c0, 3)                                                      # 1/1
        self.enc1 = nn.Sequential(DSConv(c0, c1, stride=2),
                                  *[DSResBlock(c1) for _ in range(depths[0])])               # 1/2
        self.enc2 = nn.Sequential(DSConv(c1, c2, stride=2),
                                  *[DSResBlock(c2) for _ in range(depths[1])])               # 1/4
        self.enc3 = nn.Sequential(DSConv(c2, c3, stride=2),
                                  *[DSResBlock(c3, dilation=d) for d in dilations],
                                  GlobalContext(c3) if global_context else nn.Identity())    # 1/8

        self.dec2 = Up(c3, c2, c2, use_skip=skips)                                           # 1/4
        self.dec1 = Up(c2, c1, c1, use_skip=skips)                                           # 1/2
        self.dec0 = Up(c1, c0, c0, use_skip=skips)                                           # 1/1
        self.head = nn.Conv2d(c0, 1, kernel_size=1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        # Start every residual block as an identity mapping: easier to optimize from scratch.
        for m in self.modules():
            if isinstance(m, DSResBlock):
                nn.init.zeros_(m.conv[1][1].weight)

    def forward(self, x):
        x = (x - self.mean) / self.std
        s0 = self.stem(x)
        s1 = self.enc1(s0)
        s2 = self.enc2(s1)
        x = self.enc3(s2)
        x = self.dec2(x, s2)
        x = self.dec1(x, s1)
        x = self.dec0(x, s0)
        return self.head(x)


def build_model(name="tinylanenet", **kwargs):
    """Model factory so checkpoints can record which network (and settings) they hold."""
    if name == "tinylanenet":
        return TinyLaneNet(**kwargs)
    if name == "unet_baseline":
        from baseline_unet import UNet
        return UNet(**kwargs)
    raise ValueError(f"Unknown model: {name}")


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


if __name__ == "__main__":
    model = TinyLaneNet()
    dummy = torch.rand(2, 3, 72, 128)
    out = model(dummy)
    print(f"Input:  {tuple(dummy.shape)}")
    print(f"Output: {tuple(out.shape)}")
    print(f"Params: {count_parameters(model):,}")
