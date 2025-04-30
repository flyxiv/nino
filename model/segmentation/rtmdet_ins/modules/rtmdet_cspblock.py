"""Implementation of the RTMDet CSP Block layer
Explained in the paper: https://arxiv.org/pdf/2212.07784v2 
"""

from .rtmdet_bottleneck import RTMDetBottleneck
import torch
import torch.nn as nn

class RTMDetCSPBlock(nn.Module):
    """RTMDet CSP Block Layer 

    Uses rtmdet bottleneck layer.

    Input:
        x: torch.Tensor, shape: (B, C, H, W)

    Output:
        x: torch.Tensor, shape: (B, C', H', W')
    """

    def __init__(self, in_channels, out_channels, downsample=False):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        if downsample:
            self.conv = nn.Conv2d(self.in_channels, self.in_channels, kernel_size=3, stride=2, padding=1)
        else:
            self.conv = nn.Conv2d(self.in_channels, self.in_channels, kernel_size=1, stride=1)

        self.bottleneck = RTMDetBottleneck(in_channels=in_channels)
        self.merge_conv = nn.Conv2d(self.in_channels * 2, self.out_channels, kernel_size=1, stride=1)

    def forward(self, x):
        x = self.conv(x)
        x_bottleneck = self.bottleneck(x)
        x_concat = torch.cat([x, x_bottleneck], dim=1)
        x_merged = self.merge_conv(x_concat)

        return x_merged

