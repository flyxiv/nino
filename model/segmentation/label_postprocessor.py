"""Handles postprocessing of labels for RTGDET model

Use SimOTA Algorithm to calculate cost matrix
"""

from util.label_calculation import calculate_iou
import torch
from torch.nn import BCEWithLogitsLoss
import torch.nn.functional as F
import numpy as np

EPSILON = 1e-10

class LabelPostprocessor:
    def __init__(self, num_classes, q, lambda1=1, lambda2=3, lambda3=1, alpha=10, beta=3, device='cpu'):
        self.num_classes = num_classes
        self.q = q 
        self.lambda1 = torch.tensor(lambda1, device=device)
        self.lambda2 = torch.tensor(lambda2, device=device)
        self.lambda3 = torch.tensor(lambda3, device=device)
        self.alpha = torch.tensor(alpha, device=device)
        self.beta = torch.tensor(beta, device=device)
        self.epsilon = torch.tensor(EPSILON, device=device)
        self.device = device

    def _calculate_cost_matrix_one_sample(self, cls_scores, bbox_preds, gt_labels, gt_bboxes):
        """Calculate cost matrix for one sample
        cost: lambda1 * classification loss + lambda2 * iou loss + lambda3 * center loss 
        
        classification loss: Uses soft label y_soft(using IoU between gt and pred bboxes) and prediction value y_pred
            * y_soft = IoU(pred_bbox, gt_bbox)
            * y_pred = cls_scores
            * loss = Cross Entropy(y_soft, y_pred) * (y_soft - y_pred) ** 2 

        iou loss: Uses IoU between gt and pred bboxes
            * loss = -log(IoU(pred_bbox, gt_bbox))

        center loss: Uses center distance between gt and pred bboxes
            * loss = alpha**(abs(x_pred - x_gt) - beta)

        Args: B = batch size, P = number of predictions, G = number of ground truth, C = number of classes
            cls_scores: (P, C)
            bbox_preds: (P, 4)
            gt_labels: (G)
            gt_bboxes: (G, 4)

        Returns:
            cost_matrix: (G + 1, P), one for background
            gt_pred_iou_matrix: (G, P)
        """
        num_gts = len(gt_labels)
        num_preds = len(bbox_preds)

        # IoU 계산을 위한 bbox 확장
        gt_bboxes = gt_bboxes.unsqueeze(1).expand(num_gts, num_preds, 4)  # (num_gts, num_preds, 4)
        bbox_preds = bbox_preds.unsqueeze(0).expand(num_gts, num_preds, 4)  # (num_gts, num_preds, 4)
        
        # IoU matrix 계산 (vectorized)
        # bbox format: (x_center, y_center, w, h)
        x1_pred = bbox_preds[..., 0] - bbox_preds[..., 2] / 2
        y1_pred = bbox_preds[..., 1] - bbox_preds[..., 3] / 2
        x2_pred = bbox_preds[..., 0] + bbox_preds[..., 2] / 2
        y2_pred = bbox_preds[..., 1] + bbox_preds[..., 3] / 2
        
        x1_gt = gt_bboxes[..., 0] - gt_bboxes[..., 2] / 2
        y1_gt = gt_bboxes[..., 1] - gt_bboxes[..., 3] / 2
        x2_gt = gt_bboxes[..., 0] + gt_bboxes[..., 2] / 2
        y2_gt = gt_bboxes[..., 1] + gt_bboxes[..., 3] / 2
        
        # Intersection 좌표
        x1_inter = torch.max(x1_pred, x1_gt)
        y1_inter = torch.max(y1_pred, y1_gt)
        x2_inter = torch.min(x2_pred, x2_gt)
        y2_inter = torch.min(y2_pred, y2_gt)
        
        # Intersection 면적
        w_inter = (x2_inter - x1_inter).clamp(min=0)
        h_inter = (y2_inter - y1_inter).clamp(min=0)
        inter = w_inter * h_inter
        
        # Union 면적
        area_pred = bbox_preds[..., 2] * bbox_preds[..., 3]
        area_gt = gt_bboxes[..., 2] * gt_bboxes[..., 3]
        union = area_pred + area_gt - inter
        
        # IoU matrix: (num_gts, num_preds)
        iou_matrix = inter / (union + EPSILON)
        
        # Classification cost
        gt_labels_one_hot = F.one_hot(gt_labels, num_classes=cls_scores.shape[1])  # (num_gts, num_classes)
        pred_scores = cls_scores.unsqueeze(0).expand(num_gts, num_preds, -1)  # (num_gts, num_preds, num_classes)
        
        # 각 gt에 대한 예측 점수 선택
        pred_scores_for_gt = torch.gather(pred_scores, 2, 
                                        gt_labels.view(num_gts, 1, 1).expand(num_gts, num_preds, 1))  # (num_gts, num_preds, 1)
        pred_scores_for_gt = pred_scores_for_gt.squeeze(-1)  # (num_gts, num_preds)
        
        # Classification cost 계산
        cls_cost = F.binary_cross_entropy_with_logits(
            pred_scores_for_gt,
            iou_matrix,
            reduction='none'
        ) * (iou_matrix - torch.sigmoid(pred_scores_for_gt)) ** 2
        
        # IoU cost
        iou_cost = -torch.log(iou_matrix + EPSILON)
        
        # Center cost
        pred_centers = bbox_preds[..., :2]  # (num_gts, num_preds, 2)
        gt_centers = gt_bboxes[..., :2]    # (num_gts, num_preds, 2)
        
        # L2 distance between centers
        center_dist = torch.norm(pred_centers - gt_centers, dim=2)  # (num_gts, num_preds)
        center_cost = self.alpha ** (center_dist - self.beta)
        
        # final cost matrix without background
        cost_matrix = (
            self.lambda1 * cls_cost + 
            self.lambda2 * iou_cost + 
            self.lambda3 * center_cost
        )

        num_bboxes = cost_matrix.shape[1]

        background_gt = torch.zeros_like((cost_matrix[0]))
        background_preds = cls_scores[:, -1]
        background_cost = F.binary_cross_entropy_with_logits(
            background_preds,
            background_gt,
            reduction='none'
        ) * (background_gt - torch.sigmoid(background_preds)) ** 2

        cost_matrix = torch.cat((background_cost.unsqueeze(0), cost_matrix), dim=0)

        return cost_matrix, iou_matrix

    def _dynamic_k_matching(self, cost_matrix, iou_matrix, valid_masks=None):
        """Dynamically decides the appropriate number k for each ground truth and collects top k predictions.

        Algorithm is described in 
        paper: https://openaccess.thecvf.com/content/CVPR2021/papers/Ge_OTA_Optimal_Transport_Assignment_for_Object_Detection_CVPR_2021_paper.pdf        

        But RTMDet uses a simplified version to make the calculation faster in exchange for accuracy.

        args: G is the number of ground truth, P is the number of predictions
            cost_matrix: (G + 1, P), one for background
            iou_matrix: (G, P)
            valid_masks: (G), bboxes that aren't too small

        Returns:
            matching_matrix: (G + 1, P), one for background
        """
        num_bboxes = torch.tensor([cost_matrix.shape[1]], device=self.device)

        matching_matrix = torch.zeros_like(cost_matrix)

        n_candidate_k = min(self.q, cost_matrix.shape[1])

        # find top k predictions with highest iou for each ground truth
        # topk_ious: (G, n_candidate_k)
        topk_ious, _ = torch.topk(iou_matrix, n_candidate_k, dim=1)

        # get dynamic_k = the floor of the sum of ious for each ground truth: (G)
        dynamic_ks = torch.clamp(topk_ious.sum(1).int(), min=1)
        total_matched_anchors = torch.sum(dynamic_ks)
        background_number = num_bboxes - total_matched_anchors
        dynamic_ks = torch.cat((dynamic_ks, background_number))

        # find top k predictions with highest cost for each ground truth
        for gt_idx in range(cost_matrix.shape[0]):
            _, pred_idx = torch.topk(cost_matrix[gt_idx], k=dynamic_ks[gt_idx], largest=False)
            matching_matrix[gt_idx][pred_idx] = 1

        if valid_masks is not None:
            matching_matrix = matching_matrix[:, valid_masks] 
        
        anchor_matching_gt = matching_matrix.sum(0)

        # assign bboxes that are assigned to multiple ground truths to the one with highest iou
        if (anchor_matching_gt > 1).sum() > 0:
            multiple_match_mask = anchor_matching_gt > 1
            _, cost_max = torch.max(cost_matrix[:, multiple_match_mask], dim=0)

            matching_matrix[:, multiple_match_mask] = 0
            matching_matrix[cost_max, multiple_match_mask] = 1

        matching_matrix_non_background = matching_matrix[:-1]
        matching_matrix_background = matching_matrix[-1]

        return matching_matrix_non_background, matching_matrix_background


    def calculate_cost_matrix(self, cls_scores, bbox_preds, gt_labels, gt_bboxes):
        """Calculate cost matrix for SimOTA Algorithm

        Algorithm details are described in the paper: 
           https://openaccess.thecvf.com/content/CVPR2021/papers/Ge_OTA_Optimal_Transport_Assignment_for_Object_Detection_CVPR_2021_paper.pdf

        Args: B = batch size, P = number of predictions, G = number of ground truth, C = number of classes
            cls_scores: (P, C)
            bbox_preds: (P, 4) - box predictions are in [x, y, w, h] format (x, y is bbox center)
            gt_labels: (G)
            gt_bboxes: (G, 4) - box ground truth are in [x, y, w, h] format (x, y is bbox center)

        Returns:
            matching_matrix_non_background: (G, P)
            matching_matrix_background: (P)
        """
        cost_matrix, iou_matrix = self._calculate_cost_matrix_one_sample(cls_scores, bbox_preds, gt_labels, gt_bboxes)
        matching_matrix_non_background, matching_matrix_background = self._dynamic_k_matching(cost_matrix, iou_matrix)

        return matching_matrix_non_background, matching_matrix_background

    def extract_matched_pairs(self, *, matching_matrix, matching_matrix_background, cls_scores, pred_bboxes, indices, mask_kernels, mask_feats, gt_labels, gt_bboxes, gt_masks, img_size):
        """Extract only matched (pred, gt) pairs
        Args: G is the number of ground truth, P is the number of predictions
            matching_matrix: (G, P)
            matching_matrix_background: (P)
            cls_scores: (P, C)
            pred_bboxes: (P, 4)
            indices: (P, 3)
                contains, stride, y_idx, x_idx of the grid the i-th bbox prediction is from
            mask_kernels: (LEVEL, 169, H, W)
            mask_feats: (LEVEL, MASK_FEATURES + 2, H, W)
            gt_labels: (G, 1)
            gt_bboxes: (G, 4)
            gt_masks: (G, H, W)
        
        Returns: M is the number of matched pairs
            background_preds: (BG_P)
            matched_classes: (M)
            matched_pred_bboxes: (M, 4)
            matched_gt_bboxes: (M, 4)
            matched_masks: list(tensor(M, H', W')), where H' and W' are the height and width of the mask output. 
            resized_gt_masks: list(tensor((M, H', W'))), where H' and W' are the height and width of the resized ground truth mask
        """
        matched_indices = torch.nonzero(matching_matrix, as_tuple=False)

        background_matched_indices = torch.nonzero(matching_matrix_background, as_tuple=False)
        background_preds = cls_scores[background_matched_indices, -1]
        
        gt_indices = matched_indices[:, 0]
        pt_indices = matched_indices[:, 1]

        gt_classes = gt_labels[gt_indices]
        matched_pred_classes = cls_scores[pt_indices, gt_classes]
        
        matched_gt_bboxes = gt_bboxes[gt_indices]
        matched_pred_bboxes = pred_bboxes[pt_indices]
        
        matched_bboxes_indices = indices[pt_indices]

        masks_output = []

        for (level, grid_y, grid_x) in matched_bboxes_indices:
            mask_kernel = mask_kernels[level][:, grid_y, grid_x].reshape((169))

            # (output_channel, input_channel, kernel_size, kernel_size)
            mask_kernel1_weight = mask_kernel[:80].reshape((8, 10, 1, 1))
            mask_kernel1_bias = mask_kernel[80:88].reshape((8))

            mask_kernel2 = mask_kernel[88:160].reshape((1, 8, 3, 3))
            mask_kernel3 = mask_kernel[160:].reshape((1, 1, 3, 3))

            mask_feat = mask_feats[level]

            dynamic_conv_1 = F.conv2d(mask_feat, mask_kernel1_weight, bias=mask_kernel1_bias)
            dynamic_conv_2 = F.conv2d(dynamic_conv_1, mask_kernel2, padding=1)
            dynamic_conv_3 = F.conv2d(dynamic_conv_2, mask_kernel3, padding=1)

            dynamic_conv_3 = torch.sigmoid(dynamic_conv_3)

            masks_output.append(dynamic_conv_3)

        matched_gt_masks = gt_masks[gt_indices]
        resized_gt_masks = []

        for i in range(len(matched_gt_bboxes)):
            x_center, y_center, w, h = map(int, matched_gt_bboxes[i])

            assert h > 0 and w > 0, f"more than one of h = {h}, w = {w} is negative"

            x1 = int(max(x_center - w / 2, 0))
            y1 = int(max(y_center - h / 2, 0))
            x2 = int(min(x_center + w / 2, img_size))
            y2 = int(min(y_center + h / 2, img_size))

            gt_mask_in_bbox = matched_gt_masks[i][y1:y2, x1:x2].float()
            gt_mask_interpolated_to_mask_output_size = F.interpolate(
                gt_mask_in_bbox.reshape((1, 1, *gt_mask_in_bbox.shape)),
                size=masks_output[i].shape[-2:],
                mode='bilinear',
                align_corners=False
            ).resize(*masks_output[i].shape)

            resized_gt_masks.append((gt_mask_interpolated_to_mask_output_size > 0.5).float())
        
        
        for matched_mask, resized_gt_mask in zip(masks_output, resized_gt_masks):
            assert matched_mask.shape == resized_gt_mask.shape, f"matched_mask.shape = {matched_mask.shape}, resized_gt_mask.shape = {resized_gt_mask.shape}"

        return background_preds, matched_pred_classes, matched_pred_bboxes, matched_gt_bboxes, masks_output, resized_gt_masks

        