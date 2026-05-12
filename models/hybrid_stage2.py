"""
Stage 2 model: hybrid CNN-Transformer for tumor segmentation.

  ResNet-34 encoder  [B, 512, H/32, W/32]
       -> Transformer bottleneck (or conv bottleneck for ablation)
       -> U-Net decoder with skip connections from CNN encoder
"""
import torch
import torch.nn as nn
import timm

from models.unet_stage1 import conv_bn_relu, DecoderBlock


class TransformerBottleneck(nn.Module):
    """Tokenize feature map -> self-attention layers -> de-tokenize."""

    def __init__(self, channels: int, num_layers: int = 3, num_heads: int = 8,
                 dim_feedforward: int = 1024, dropout: float = 0.1,
                 max_tokens: int = 256):
        super().__init__()
        self.channels = channels

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=channels, nhead=num_heads,
            dim_feedforward=dim_feedforward, dropout=dropout,
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pos_embed = nn.Parameter(torch.zeros(1, max_tokens, channels))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.max_tokens = max_tokens

    def forward(self, x):
        B, C, H, W = x.shape
        n_tokens = H * W
        assert n_tokens <= self.max_tokens, (
            f"Got {n_tokens} tokens but pos_embed sized for {self.max_tokens}. "
            f"Increase max_tokens or reduce input size."
        )
        tokens = x.flatten(2).transpose(1, 2)
        tokens = tokens + self.pos_embed[:, :n_tokens, :]
        tokens = self.transformer(tokens)
        return tokens.transpose(1, 2).reshape(B, C, H, W)


class HybridSegmenter(nn.Module):
    """CNN encoder + optional transformer bottleneck + U-Net decoder."""

    def __init__(self,
                 encoder_name: str = "resnet34", pretrained: bool = True,
                 use_transformer: bool = True,
                 transformer_layers: int = 3, transformer_heads: int = 8,
                 transformer_dim_feedforward: int = 1024,
                 transformer_dropout: float = 0.1,
                 input_size: int = 256, num_classes: int = 1):
        super().__init__()
        self.use_transformer = use_transformer

        self.encoder = timm.create_model(
            encoder_name, pretrained=pretrained,
            features_only=True, in_chans=3,
            out_indices=(0, 1, 2, 3, 4),
        )
        enc_channels = self.encoder.feature_info.channels()
        c0, c1, c2, c3, c4 = enc_channels

        if use_transformer:
            deep_hw = max(input_size // 32, 1)
            max_tokens = max(deep_hw * deep_hw, 64)
            self.bottleneck = TransformerBottleneck(
                channels=c4, num_layers=transformer_layers,
                num_heads=transformer_heads,
                dim_feedforward=transformer_dim_feedforward,
                dropout=transformer_dropout, max_tokens=max_tokens,
            )
        else:
            self.bottleneck = nn.Sequential(
                conv_bn_relu(c4, c4),
                conv_bn_relu(c4, c4),
            )

        self.dec4 = DecoderBlock(c4, c3, 256)
        self.dec3 = DecoderBlock(256, c2, 128)
        self.dec2 = DecoderBlock(128, c1, 64)
        self.dec1 = DecoderBlock(64, c0, 32)
        self.dec0 = DecoderBlock(32, 0, 16)
        self.final = nn.Conv2d(16, num_classes, kernel_size=1)

    def forward(self, x):
        feats = self.encoder(x)
        f0, f1, f2, f3, f4 = feats
        x = self.bottleneck(f4)
        x = self.dec4(x, f3)
        x = self.dec3(x, f2)
        x = self.dec2(x, f1)
        x = self.dec1(x, f0)
        x = self.dec0(x, None)
        return self.final(x)
