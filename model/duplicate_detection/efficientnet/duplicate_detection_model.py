"""Use EfficientNet with triple head to detect clarity of each image and similarity of two images"""

import torch
import torch.nn as nn
import torchvision

class DuplicateDetectionModel(nn.Module):
    def __init__(self):
        super(DuplicateDetectionModel, self).__init__()

        self.efficientnet = torchvision.models.efficientnet_v2_l(weights=torchvision.models.EfficientNet_V2_L_Weights.DEFAULT)
        self.num_bottleneck_features = self.efficientnet.classifier[-1].in_features
        self.efficientnet.classifier = nn.Identity()
        self.clarity_layer = nn.Linear(self.num_bottleneck_features, 1)
        self.similarity_layer = nn.Linear(self.num_bottleneck_features * 2, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, image1, image2):
        image1_features = self.efficientnet(image1)
        image2_features = self.efficientnet(image2)

        image1_clarity = self.clarity_layer(image1_features)
        image1_clarity = self.sigmoid(image1_clarity)

        image2_clarity = self.clarity_layer(image2_features)
        image2_clarity = self.sigmoid(image2_clarity)

        similarity = self.similarity_layer(torch.cat((image1_features, image2_features), dim=1))
        similarity = self.sigmoid(similarity)

        return image1_clarity, image2_clarity, similarity


