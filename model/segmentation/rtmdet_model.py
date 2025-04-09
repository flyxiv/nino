"""Main class for RTMDET model, which will be used for segmenting 2d pixel sprites
"""

import torch
from torch import nn
from .modules.rtmdet_backbone import RTMDETBackbone
from .modules.pafpn import PAFPN 
from .modules.rtmdet_head import RTMDetHead 

P3_STRIDE = 8

class RTMDETModel(nn.Module):
    def __init__(self, num_classes, img_size=1024, init_channels=64, device='cpu'):
        super().__init__()
        self.img_size = img_size
        self.pafpn_in_channels = init_channels * 4
        self.pafpn_out_channels = init_channels * 4

        self.num_classes = num_classes
        self.backbone = RTMDETBackbone(img_size, init_channels, device)
        self.pafpn = PAFPN(in_channels=self.pafpn_in_channels, out_channels=self.pafpn_out_channels)
        self.head = RTMDetHead(in_channels=self.pafpn_out_channels, num_classes=num_classes, device=device)
        self.device = device

        self.to(device)

    def forward(self, x):
        _, _, c3, c4, c5 = self.backbone(x)
        p3, p4, p5 = self.pafpn(c3, c4, c5)
        cls_scores, bbox_preds, indices, p3_mask_kernels, p3_mask_feats, p4_mask_kernels, p4_mask_feats, p5_mask_kernels, p5_mask_feats = self.head(p3, p4, p5)

        assert len(cls_scores.shape) == len(bbox_preds.shape) == len(indices.shape)
        assert cls_scores.shape[1] == bbox_preds.shape[1] == indices.shape[1]

        return cls_scores, self._decode_bbox_outputs(bbox_preds, indices), indices, p3_mask_kernels, p3_mask_feats, p4_mask_kernels, p4_mask_feats, p5_mask_kernels, p5_mask_feats

    def _decode_bbox_outputs(self, bbox_preds, indices):
        """Rescale bbox predictions to the original image size

        The bbox predictions need to be decoded because of two reasons:
        1) the bbox coordinates are estimated as offset from the grid cell, not from the image origin
        2) the bbox width/height are estimated as log scale, so they need to be exponentiated
        
        Args: B is the batch size, P is the number of predictions, 8 is the number of bbox coordinates
            bbox_preds: [B, P, 4] 
                each bbox_preds_pi is a tensor of shape (B, 4, H_i, W_i) where H_i and W_i are the height and width of the i-th feature map

            indices: [B, P, 3]
                contains stride, y_idx, x_idx of the grid the i-th bbox prediction is from

        Outputs:
            bbox_preds_decoded = all decoded bbox predictions concated in [B, P, 4] format
        """

        decoded_bboxes = []

        for (bbox_pred, idx_data) in zip(bbox_preds, indices):
            xy_offset = torch.nn.functional.sigmoid(bbox_pred[..., :2])

            stride = torch.pow(2, 3 + idx_data[..., 0])

            x_center = (idx_data[..., 2] + xy_offset[..., 0]) * stride  
            y_center = (idx_data[..., 1] + xy_offset[..., 1]) * stride  
            
            w = torch.exp(bbox_pred[..., 2]) * stride
            h = torch.exp(bbox_pred[..., 3]) * stride

            decoded_bbox = torch.stack([x_center, y_center, w, h], dim=-1)
            decoded_bboxes.append(decoded_bbox.unsqueeze(0))
        
        bbox_preds_decoded = torch.cat(decoded_bboxes, dim=0)

        return bbox_preds_decoded
