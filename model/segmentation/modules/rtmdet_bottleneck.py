"""Implementation of the RTMDet Bottleneck Block layer
Explained in the paper: https://arxiv.org/pdf/2212.07784v2 
"""

from torch import nn

class RTMDetBottleneck(nn.Module):
    """RTMDet Bottleneck Block Layer 

    Uses 5x5 depthwise Convolution layer and a Pointwise 1x1 Convolution layer
    Preserves 
    """
    def __init__(self, in_channels):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = in_channels

        self.conv1 = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=3, stride=1, padding=1)

        self.batch_norm1 = nn.BatchNorm2d(self.out_channels)
        self.silu = nn.SiLU()

        # groups=self.out_channels for depthwise convolution
        self.conv2 = nn.Sequential(
            nn.Conv2d(self.out_channels, self.out_channels, kernel_size=5, stride=1, groups=self.out_channels, padding=2),
            nn.Conv2d(self.out_channels, self.out_channels, kernel_size=1, stride=1),
        )
        self.batch_norm2 = nn.BatchNorm2d(self.out_channels)

    def forward(self, x):
        x = self.conv1(x)
        x = self.batch_norm1(x)
        x = self.silu(x)

        x = self.conv2(x)
        x = self.batch_norm2(x)
        x = self.silu(x)
        return x

