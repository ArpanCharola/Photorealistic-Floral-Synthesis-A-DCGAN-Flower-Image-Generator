"""Conditional DCGAN generator (class-conditioned, size-parametric).

Upgrades the original unconditional generator in two ways:
  1. A learned class embedding is fused with the noise vector so a chosen
     species can be requested (label-embedding concatenation cGAN).
  2. The conv stack is built from ``img_size`` (64 or 128) instead of being
     hard-coded, so the same class trains at either resolution.
"""

import math

import torch
import torch.nn as nn


def _g_block(in_ch, out_ch, kernel=4, stride=2, padding=1):
    """ConvTranspose -> BatchNorm -> ReLU (a single upsampling stage)."""
    return nn.Sequential(
        nn.ConvTranspose2d(in_ch, out_ch, kernel, stride, padding, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(True),
    )


class Generator(nn.Module):
    def __init__(self, z_dim=100, embed_dim=50, num_classes=102,
                 img_channels=3, img_size=64, feature_maps=64):
        super().__init__()
        self.z_dim = z_dim
        self.embed_dim = embed_dim
        self.num_classes = num_classes

        # Learned per-class vector. Kept separate from the conv init
        # (weights_init only touches Conv/BatchNorm), so it uses default init.
        self.label_emb = nn.Embedding(num_classes, embed_dim)

        fm = feature_maps
        in_ch = z_dim + embed_dim            # noise + class code fused together

        # Number of stride-2 stages to grow 4x4 -> img_size (incl. output layer).
        num_up = int(math.log2(img_size // 4))
        if 4 * (2 ** num_up) != img_size:
            raise ValueError("img_size must be 4 * 2^k (e.g. 64 or 128)")

        layers = [
            # 1x1 -> 4x4 (project the fused code up to a small spatial map)
            _g_block(in_ch, fm * 8, kernel=4, stride=1, padding=0),
        ]
        # Middle stages, each doubling spatial size and halving channels.
        ch = fm * 8
        for i in range(1, num_up):
            out_ch = (fm * 8) // (2 ** i)
            layers.append(_g_block(ch, out_ch))
            ch = out_ch
        # Final stage -> RGB in [-1, 1].
        layers.append(nn.ConvTranspose2d(ch, img_channels, 4, 2, 1, bias=False))
        layers.append(nn.Tanh())
        self.net = nn.Sequential(*layers)

    # -------------------------------------------------------------------
    # ★ CONTRIBUTION POINT (conditioning fusion) — see project notes.
    # This is the design decision that makes the GAN steerable: how the
    # class embedding combines with the noise. The default below is
    # concatenation (the textbook cGAN choice). Trade-off to consider:
    #   embed_dim too small -> classes blur together (weak conditioning);
    #   embed_dim too large  -> swamps the noise -> less variety per class.
    # Alternative worth trying: additive fusion (project emb to z_dim, then
    # z + emb) instead of concat. Change `in_ch` above to z_dim if you do.
    # -------------------------------------------------------------------
    def forward(self, z, labels):
        # z: (B, z_dim)   labels: (B,) int64 in [0, num_classes)
        emb = self.label_emb(labels)                 # (B, embed_dim)
        fused = torch.cat([z, emb], dim=1)           # (B, z_dim + embed_dim)
        fused = fused.unsqueeze(-1).unsqueeze(-1)     # (B, C, 1, 1)
        return self.net(fused)
