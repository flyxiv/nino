"""RTMDET Backbone module

Variable names are from the model paper: https://arxiv.org/abs/2203.16519
"""

from torch import nn
from .rtmdet_cspblock import RTMDetCSPBlock

class RTMDetBackbone(nn.Module):
    def __init__(self, img_size=1024, init_channels=16, device='cpu'):
        super().__init__()
        self.img_size = img_size
        self.init_channels = init_channels
        self.device = device

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, self.init_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(self.init_channels),
            nn.SiLU(inplace=True),
            RTMDetCSPBlock(self.init_channels, self.init_channels).to(self.device),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(self.init_channels, self.init_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(self.init_channels),
            nn.SiLU(inplace=True),
            RTMDetCSPBlock(self.init_channels, self.init_channels).to(self.device),
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(self.init_channels, self.init_channels * 2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(self.init_channels * 2),
            nn.SiLU(inplace=True),
            RTMDetCSPBlock(self.init_channels * 2, self.init_channels * 2).to(self.device),
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(self.init_channels * 2, self.init_channels * 4, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(self.init_channels * 4),
            nn.SiLU(inplace=True),
            RTMDetCSPBlock(self.init_channels * 4, self.init_channels * 4).to(self.device),
        )
        
        self.conv5 = nn.Sequential(
            nn.Conv2d(self.init_channels * 4, self.init_channels * 8, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(self.init_channels * 8),
            nn.SiLU(inplace=True),
            RTMDetCSPBlock(self.init_channels * 8, self.init_channels * 8).to(self.device),
        )

        
    def forward(self, x):
        c1 = self.conv1(x)
        assert c1.shape == (x.shape[0], self.init_channels, x.shape[2] // 2, x.shape[3] // 2), f"c1.shape: {c1.shape}, x.shape: {x.shape}"
        c2 = self.conv2(c1)
        assert c2.shape == (x.shape[0], self.init_channels, x.shape[2] // 4, x.shape[3] // 4), f"c2.shape: {c2.shape}, x.shape: {x.shape}"
        c3 = self.conv3(c2)
        assert c3.shape == (x.shape[0], self.init_channels * 2, x.shape[2] // 8, x.shape[3] // 8), f"c3.shape: {c3.shape}, x.shape: {x.shape}"
        c4 = self.conv4(c3)
        assert c4.shape == (x.shape[0], self.init_channels * 4, x.shape[2] // 16, x.shape[3] // 16), f"c4.shape: {c4.shape}, x.shape: {x.shape}"
        c5 = self.conv5(c4)
        assert c5.shape == (x.shape[0], self.init_channels * 8, x.shape[2] // 32, x.shape[3] // 32), f"c5.shape: {c5.shape}, x.shape: {x.shape}"

        return c1, c2, c3, c4, c5

