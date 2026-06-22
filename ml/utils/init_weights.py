"""DCGAN weight initialization (Radford et al., 2015).

Conv / ConvTranspose weights ~ N(0, 0.02); BatchNorm weights ~ N(1, 0.02),
bias = 0. The classname filter intentionally does NOT match ``nn.Embedding``,
so the class-conditioning embeddings keep PyTorch's default init.
"""

import torch.nn as nn


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
