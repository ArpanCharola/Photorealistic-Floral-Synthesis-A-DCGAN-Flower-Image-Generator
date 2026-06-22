"""Conditional DCGAN discriminator (class-conditioned, size-parametric).

Two changes from the original:
  1. The class label is injected as an extra image-sized channel, so the
     discriminator judges "is this a real <species>?" not just "is this real?".
  2. The output is a raw logit (no Sigmoid) — paired with BCEWithLogitsLoss
     in training, which is numerically stabler and lets us do label smoothing.
"""

import math

import torch
import torch.nn as nn


def _d_block(in_ch, out_ch, batch_norm=True):
    """Conv (downsample) -> optional BatchNorm -> LeakyReLU."""
    layers = [nn.Conv2d(in_ch, out_ch, 4, 2, 1, bias=False)]
    if batch_norm:
        layers.append(nn.BatchNorm2d(out_ch))
    layers.append(nn.LeakyReLU(0.2, inplace=True))
    return nn.Sequential(*layers)


class Discriminator(nn.Module):
    def __init__(self, num_classes=102, img_channels=3, img_size=64,
                 feature_maps=64):
        super().__init__()
        self.img_size = img_size

        # One scalar per pixel per class -> reshaped to a (1, H, W) label map.
        self.label_emb = nn.Embedding(num_classes, img_size * img_size)

        fm = feature_maps
        in_ch = img_channels + 1             # image + 1 label-map channel

        num_down = int(math.log2(img_size // 4))
        if 4 * (2 ** num_down) != img_size:
            raise ValueError("img_size must be 4 * 2^k (e.g. 64 or 128)")

        # First conv: no BatchNorm (DCGAN convention).
        layers = [_d_block(in_ch, fm, batch_norm=False)]
        ch = fm
        for i in range(1, num_down):
            out_ch = fm * (2 ** i)
            layers.append(_d_block(ch, out_ch))
            ch = out_ch
        # Final conv -> single logit at 1x1.
        layers.append(nn.Conv2d(ch, 1, 4, 1, 0, bias=False))
        self.net = nn.Sequential(*layers)

    def forward(self, x, labels):
        # x: (B, 3, H, W)   labels: (B,) int64
        b = x.size(0)
        label_map = self.label_emb(labels).view(b, 1, self.img_size, self.img_size)
        inp = torch.cat([x, label_map], dim=1)        # (B, 4, H, W)
        return self.net(inp).view(b)                  # (B,) logits
