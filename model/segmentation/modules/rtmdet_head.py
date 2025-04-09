"""Implements RTMDet Head for object detection and instance segmentation
Based on Paper: https://arxiv.org/pdf/2212.07784v2
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from util.label_calculation import create_coord_features

class RTMDetHead(nn.Module):
    """RTMDet Head for object detection and instance segmentation
    
    Contains:
    1. Classification Head
    2. Regression Head
    3. Mask Head (for instance segmentation)
    """

    MASK_FEATURES = 8
    MASK_KERNEL_FEATURES = 169
    
    def __init__(self, 
                 in_channels,       
                 feat_channels=128, 
                 num_classes=80,    
                 stacked_convs=2,   
                 device='cpu'):  
        super().__init__()
        self.device = device
        
        self.in_channels = in_channels
        self.feat_channels = feat_channels
        self.num_classes = num_classes
        self.stacked_convs = stacked_convs
        
        # Classification Head
        cls_convs = []
        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            cls_convs.append(
                nn.Conv2d(chn, self.feat_channels, kernel_size=3, padding=1))
            cls_convs.append(nn.BatchNorm2d(self.feat_channels))
            cls_convs.append(nn.SiLU(inplace=True))
        self.cls_convs = nn.Sequential(*cls_convs)
        self.cls_out = nn.Conv2d(self.feat_channels, self.num_classes, kernel_size=1)
        self.softmax = nn.Softmax(dim=2)
        
        # Regression Head
        reg_convs = []
        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            reg_convs.append(
                nn.Conv2d(chn, self.feat_channels, kernel_size=3, padding=1))
            reg_convs.append(nn.BatchNorm2d(self.feat_channels))
            reg_convs.append(nn.SiLU(inplace=True))
        self.reg_convs = nn.Sequential(*reg_convs)
        self.reg_out= nn.Conv2d(self.feat_channels, 4, kernel_size=1) 

        # Mask Head
        self.mask_kernel_convs = nn.Sequential(
            nn.Conv2d(self.in_channels, self.feat_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.feat_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(self.feat_channels, self.feat_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.feat_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(self.feat_channels, self.MASK_KERNEL_FEATURES, kernel_size=1)  
        )
        
        self.mask_feat_convs = nn.Sequential(
            nn.Conv2d(self.in_channels, self.feat_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.feat_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(self.feat_channels, self.feat_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.feat_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(self.feat_channels, self.feat_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.feat_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(self.feat_channels, self.feat_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.feat_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(self.feat_channels, self.MASK_FEATURES, kernel_size=1)  
        )

    def _forward_single_level(self, x, level_idx):
        B, _, H, W = x.size()

        cls_feat = self.cls_convs(x)
        cls_score = self.cls_out(cls_feat)
        cls_score = cls_score.permute(0, 2, 3, 1)
        cls_score = cls_score.reshape(cls_score.size(0), -1, self.num_classes)
        cls_score = self.softmax(cls_score)

        reg_feat = self.reg_convs(x)
        bbox_pred = self.reg_out(reg_feat)
        bbox_pred = bbox_pred.permute(0, 2, 3, 1)
        bbox_pred = bbox_pred.reshape(bbox_pred.size(0), -1, 4)

        # TODO: Just create one level indices for all batch
        h_indices = torch.arange(H, device=cls_score.device).view(-1, 1).repeat(1, W).flatten()
        w_indices = torch.arange(W, device=cls_score.device).repeat(H)
        level_indices = torch.full_like(h_indices, level_idx)
        indices = torch.stack([level_indices, h_indices, w_indices], dim=1)
        batch_indices = indices.unsqueeze(0).repeat(B, 1, 1)
        
        mask_kernels = self.mask_kernel_convs(x)
        mask_feats = self.mask_feat_convs(x)

        mask_feats = torch.concat([mask_feats, create_coord_features(B, H, W, device=self.device)], dim=1)

        assert mask_feats.size(1) == self.MASK_FEATURES + 2
        assert mask_kernels.size(1) == self.MASK_KERNEL_FEATURES

        return cls_score, bbox_pred, batch_indices, mask_kernels, mask_feats

    def forward(self, p3, p4, p5):
        """Gets P3, P4, P5 input from PAFPN and gives outputs needed for creating predictions.
        
        Args:
            p3, p4, p5: outputs from PAFPN
            
        Returns:
            cls_scores: (B, H*W, n_classes) has n_classes predictions for every P3/P4/P5 pixels  

            bbox_preds: (B, H*W, 4) has 4 predictions for every P3/P4/P5 pixels, one bbox
                        prediction is of format (center_dx_from_grid, center_dy_from_grid, log(w), log(h))

                        Calculated in log scale to ensure positive values(will be exponentiated later) + smoothen bigger bbox values

            p3_masks_kernel: (B, MASK_KERNEL_FEATURES, H, W)
            p3_masks_feat: (B, MASK_FEATURES + 2, H, W)

            p4_masks_kernel: (B, MASK_KERNEL_FEATURES, H, W)
            p4_masks_feat: (B, MASK_FEATURES + 2, H, W)

            p5_masks_kernel: (B, MASK_KERNEL_FEATURES, H, W)
            p5_masks_feat: (B, MASK_FEATURES + 2, H, W)
        """
        
        p3_cls_score, p3_bbox_pred, p3_batch_indices, p3_mask_kernels, p3_mask_feats = self._forward_single_level(p3, 0)
        p4_cls_score, p4_bbox_pred, p4_batch_indices, p4_mask_kernels, p4_mask_feats = self._forward_single_level(p4, 1)
        p5_cls_score, p5_bbox_pred, p5_batch_indices, p5_mask_kernels, p5_mask_feats = self._forward_single_level(p5, 2)
            
        cls_score = torch.cat([p3_cls_score, p4_cls_score, p5_cls_score], dim=1)
        bbox_pred = torch.cat([p3_bbox_pred, p4_bbox_pred, p5_bbox_pred], dim=1)
        batch_indices = torch.cat([p3_batch_indices, p4_batch_indices, p5_batch_indices], dim=1)

        return cls_score, bbox_pred, batch_indices, p3_mask_kernels, p3_mask_feats, p4_mask_kernels, p4_mask_feats, p5_mask_kernels, p5_mask_feats
    