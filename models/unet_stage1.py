"""
Stage 1 model: U-Net with ResNet-34 encoder (ImageNet pretrained).
Segments LIVER from full CT slices.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


def conv_bn_relu(in_ch, out_ch, k=3, p=1):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, k, padding=p, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class DecoderBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.conv1 = conv_bn_relu(in_ch + skip_ch, out_ch)
        self.conv2 = conv_bn_relu(out_ch, out_ch)

    def forward(self, x, skip):
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        if skip is not None:
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear",
                                  align_corners=False)
            x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class ResNetUNet(nn.Module):
    """U-Net with timm ResNet encoder."""

    def __init__(self, encoder_name: str = "resnet34", pretrained: bool = True,
                 num_classes: int = 1):
        super().__init__()
        self.encoder = timm.create_model(
            encoder_name, pretrained=pretrained,
            features_only=True, in_chans=3,
            out_indices=(0, 1, 2, 3, 4),
        )
        enc_channels = self.encoder.feature_info.channels()
        c0, c1, c2, c3, c4 = enc_channels
        self.center = conv_bn_relu(c4, c4)

        self.dec4 = DecoderBlock(c4, c3, 256)
        self.dec3 = DecoderBlock(256, c2, 128)
        self.dec2 = DecoderBlock(128, c1, 64)
        self.dec1 = DecoderBlock(64, c0, 32)
        self.dec0 = DecoderBlock(32, 0, 16)
        self.final = nn.Conv2d(16, num_classes, kernel_size=1)

    def forward(self, x):
        feats = self.encoder(x)
        f0, f1, f2, f3, f4 = feats
        x = self.center(f4)
        x = self.dec4(x, f3)
        x = self.dec3(x, f2)
        x = self.dec2(x, f1)
        x = self.dec1(x, f0)
        x = self.dec0(x, None)
        return self.final(x)
