"""RTMDet Backbone model with classifier head to pretrain classification task with ImageNet
"""

from .modules.rtmdet_backbone import RTMDetBackbone
import torch.nn as nn
import logging
import torch
from preprocessing.imagenet_dataset import get_imagenet_dataloaders, IMAGENET_LABEL_COUNT

class RTMDetBackboneClassifier(nn.Module):
    def __init__(self, num_classes=IMAGENET_LABEL_COUNT, img_size=1024, init_channels=16, device='cpu'):
        super().__init__()
        self.backbone = RTMDetBackbone(img_size, init_channels, device)

        c5_channels = init_channels * 8
        c5_spatial_size = img_size // 32
        feature_size = c5_channels * c5_spatial_size * c5_spatial_size
        
        self.classifier = nn.Linear(feature_size, num_classes)
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x):
        x = self.backbone(x)[-1]
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        x = self.softmax(x)
        return x

    def save_checkpoint(self, path):
        torch.save(self.state_dict(), path)

    def save_backbone(self, path):
        torch.save(self.backbone.state_dict(), path)

    def load_model(self, path):
        self.load_state_dict(torch.load(path))