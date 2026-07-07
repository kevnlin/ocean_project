"""Small 2D U-Net for depthwise field reconstruction.

Operates on (B, C_in, H, W) maps; B can stack depth-slices (depthwise) so a
single shared U-Net reconstructs every depth level.  Deliberately small — this
is a baseline, not a foundation model.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1), nn.GroupNorm(8, cout), nn.SiLU(),
            nn.Conv2d(cout, cout, 3, padding=1), nn.GroupNorm(8, cout), nn.SiLU(),
        )

    def forward(self, x):
        return self.net(x)


class UNet2D(nn.Module):
    def __init__(self, c_in, c_out, base=32):
        super().__init__()
        self.inc = DoubleConv(c_in, base)
        self.d1 = DoubleConv(base, base * 2)
        self.d2 = DoubleConv(base * 2, base * 4)
        self.pool = nn.MaxPool2d(2)
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.u2 = DoubleConv(base * 4, base * 2)
        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.u1 = DoubleConv(base * 2, base)
        self.outc = nn.Conv2d(base, c_out, 1)

    def forward(self, x):
        # pad H,W to multiples of 4 (two pools)
        _, _, H, W = x.shape
        ph = (4 - H % 4) % 4
        pw = (4 - W % 4) % 4
        x = F.pad(x, (0, pw, 0, ph), mode="replicate")
        x1 = self.inc(x)
        x2 = self.d1(self.pool(x1))
        x3 = self.d2(self.pool(x2))
        y = self.up2(x3)
        y = self.u2(torch.cat([y, x2], 1))
        y = self.up1(y)
        y = self.u1(torch.cat([y, x1], 1))
        y = self.outc(y)
        return y[:, :, :H, :W]
