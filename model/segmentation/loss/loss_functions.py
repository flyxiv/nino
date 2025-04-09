"""Functions for calculating the loss of the segmentation model.
Losses:
    * Classification Loss
    * Bounding Box Loss
    * Segmentation Mask Loss
"""

import torch
from torch.nn import functional as F

def calculate_loss(background_preds, cls_preds, pred_bboxes, gt_bboxes, pred_masks, gt_masks):
    """Calculate the loss of the segmentation model.

    Args: M is the number of matched anchor boxes, C is the number of classes, W' and H' are the width and height of the mask
        cls_preds: (M) 
        pred_bboxes: (M, 4)
        gt_bboxes: (M, 4)
        pred_masks: (M, W', H')
        gt_masks: (M, W', H')
    """

    cls_background_targets = torch.zeros_like(background_preds)
    cls_background_loss = F.binary_cross_entropy_with_logits(background_preds, cls_background_targets)

    cls_targets = torch.ones_like(cls_preds)
    cls_loss = F.binary_cross_entropy_with_logits(cls_preds, cls_targets)

    bbox_loss = F.smooth_l1_loss(pred_bboxes, gt_bboxes)
    mask_loss = sum(F.binary_cross_entropy_with_logits(pred_mask, gt_mask) for pred_mask, gt_mask in zip(pred_masks, gt_masks))

    return cls_background_loss + cls_loss + bbox_loss + mask_loss
