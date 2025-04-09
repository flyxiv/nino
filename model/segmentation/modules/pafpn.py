"""Implementation of the RTMDet PAFPN Block 
Explained in the paper: https://arxiv.org/pdf/2212.07784v2 
"""

import torch
import torch.nn as nn 
import torch.nn.functional as F

from .rtmdet_cspblock import RTMDetCSPBlock

class PAFPN(nn.Module):
    """PAFPN Block
    A feature pyramid model first introduced in https://paperswithcode.com/method/pafpn

    in_channels: Channel of C3, the layer with least channel. C4, C5 have *2, *4 channels respectively.
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.c5_reduce = nn.Conv2d(in_channels=in_channels * 4, out_channels=out_channels, kernel_size=1, stride=1, padding=0)
        self.c4_reduce = nn.Conv2d(in_channels=in_channels * 2, out_channels=out_channels, kernel_size=1, stride=1, padding=0)
        self.c3_reduce = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0)

        self.c4_csp_up = RTMDetCSPBlock(in_channels=out_channels * 2, out_channels=out_channels, downsample=False)
        self.c3_csp_up = RTMDetCSPBlock(in_channels=out_channels * 2, out_channels=out_channels, downsample=False)

        self.c3_csp_down = RTMDetCSPBlock(in_channels=out_channels, out_channels=out_channels, downsample=True)
        self.c4_csp_down = RTMDetCSPBlock(in_channels=out_channels * 2, out_channels=out_channels, downsample=True)

        self.c3_out = RTMDetCSPBlock(in_channels=out_channels, out_channels=out_channels, downsample=False)
        self.c4_out = RTMDetCSPBlock(in_channels=out_channels * 2, out_channels=out_channels, downsample=False)
        self.c5_out = RTMDetCSPBlock(in_channels=out_channels * 2, out_channels=out_channels, downsample=False)


    def forward(self, c3, c4, c5):
        """Receives 3 feature maps from the CSPBackbone
        
        Input: 3 feature maps
            c3, c4, c5 all in image format (B, C, H, W)

        Output: 3 feature maps
            c3_concat, c4_concat, c5_concat all in image format (B, C, H, W)
        """
        p5 = self.c5_reduce(c5)
        p4 = self.c4_reduce(c4)
        p3 = self.c3_reduce(c3)

        # 1. Downward concatenation
        p5_up = F.interpolate(p5, size=p4.shape[2:], mode='bilinear')
        p4_plus = torch.cat([p5_up, p4], dim=1)
        p4_up = self.c4_csp_up(p4_plus)

        p4_up_up = F.interpolate(p4_up, size=p3.shape[2:], mode='bilinear')
        p3_plus = torch.cat([p4_up_up, p3], dim=1)
        p3_out_temp = self.c3_csp_up(p3_plus)

        # 2. Upward concatenation
        p3_down = self.c3_csp_down(p3_out_temp)
        p4_plus_2 = torch.cat([p3_down, p4_up], dim=1)
        p4_out_temp = self.c4_csp_down(p4_plus_2)

        p5_plus = torch.cat([p4_out_temp, p5], dim=1)

        p3_out = self.c3_out(p3_out_temp)
        p4_out = self.c4_out(p4_plus_2)
        p5_out = self.c5_out(p5_plus)

        return p3_out, p4_out, p5_out
