"""Calculations needed for label processing
"""

import torch
from collections import defaultdict
from .top_k_anchor_heap import TopKAnchorHeap

def create_coord_features(n_batch, w, h, device='cpu'):
    x_range = torch.linspace(-1, 1, w)
    y_range = torch.linspace(-1, 1, h)
    y, x = torch.meshgrid(y_range, x_range, indexing='ij')
    coord_features_batch = torch.stack([x, y], dim=0).unsqueeze(0).expand(n_batch, -1, -1, -1)

    return coord_features_batch.to(device)

def calculate_iou(bbox_pred, bbox_label):
    """
    Calculate IoU(Intersection over Union) between two bounding boxes.

    bbox value: [x, y, w, h], all values are normalized to [0, 1] + x, y are upper left corner
    
    Returns:
        float: IoU value (0~1)
    """
    x1, y1, w1, h1 = bbox_pred 
    x2, y2, w2, h2 = bbox_label

    area1 = w1 * h1
    area2 = w2 * h2
    
    x1_right = x1 + w1
    y1_bottom = y1 + h1
    x2_right = x2 + w2
    y2_bottom = y2 + h2
    
    intersection_x_left = max(x1, x2)
    intersection_y_top = max(y1, y2)
    intersection_x_right = min(x1_right, x2_right)
    intersection_y_bottom = min(y1_bottom, y2_bottom)
    
    intersection_width = max(0, intersection_x_right - intersection_x_left)
    intersection_height = max(0, intersection_y_bottom - intersection_y_top)
    
    intersection_area = intersection_width * intersection_height
    
    union_area = area1 + area2 - intersection_area
    
    iou = intersection_area / union_area if union_area > 0 else 0
    
    return iou.to(bbox_pred.device)

def calculate_segmentation_loss(cls_scores, bbox_preds, mask_preds, gt_labels, gt_bbox_labels, gt_masks, iou_negative_threshold=0.15):
    """
    Classifies each prediction into positive/ignore/negative

    positive: anchor with high enough match with ground truth(IoU)
    negative: anchors with low classification loss
    ignore: anchors with middle classification loss and not positive

    Args:
        cls_scores: (num_anchors, num_classes)
        bbox_preds: (num_anchors, 4), [bbox_upper_left_x, bbox_upper_left_y, width, height] normalized to [0, 1]
        mask_preds: (num_anchors, num_masks)
        gt_labels: (num_anchors)
        gt_bbox_labels: (num_anchors, 4), 
        gt_masks: (num_anchors, num_masks)

        * bbox_preds: [bbox_upper_left_x, bbox_upper_left_y, width, height] normalized to [0, 1]
    """

    top_k_highest_iou_anchors = dict() 
    max_iou_for_each_prediction = defaultdict(float)

    for label_idx, gt_bbox_label in enumerate(gt_bbox_labels):
        top_k_anchor_heap = TopKAnchorHeap(k=5)
        for pred_idx, bbox_pred in enumerate(bbox_preds):
            iou = calculate_iou(bbox_pred, gt_bbox_label)
            top_k_anchor_heap.add(iou, pred_idx)

            if iou > min_iou_for_each_prediction[label_idx]:
                max_iou_for_each_prediction[label_idx] = iou
            
        top_k_highest_iou_anchors[label_idx] = top_k_anchor_heap.get_top_k()


    positive_pred_indices = set()
    negative_pred_indices = set()


