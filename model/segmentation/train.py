"""Script for training RTMDET model

run ex) in nino base directory,
```sh
python -m model.segmentation.train --dataset-dir ./data/instance_segmentation_coco --batch-size 16 --epochs 10 --lr 0.0001 --device cuda --img-size 1024 --save-dir ./ --model-name rtmdet_model
```
"""

import argparse 
import logging
import torch

from tqdm import tqdm
from torch.optim import AdamW

from .input.load_coco_format_segmentation import get_coco_dataloaders 
from .rtmdet_model import RTMDETModel
from .loss.loss_functions import calculate_loss
from .label_postprocessor import LabelPostprocessor

PERFORMACE_OUTPUT_STEPS = 100 

def evaluate_model(model, valid_loader, epoch, device):
    """Evaluate RTMDET model
    """

    model.eval()

    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(valid_loader):
            cls_scores, bbox_preds, mask_preds = model(images)

            loss = calculate_loss(cls_scores, bbox_preds, mask_preds, masks, boxes)
            print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item()}")

def train_rtmdet_model(model, train_loader, valid_loader, test_loader, epochs, lr, device):
    """Train RTMDET model
    """
    optimizer = AdamW(model.parameters(), lr=lr)
    label_postprocessor = LabelPostprocessor(num_classes=model.num_classes + 1, q=5, device=device)

    if device == 'cuda':
        model = model.to(device)

    for epoch in range(epochs):
        for batch_idx, (images, targets) in tqdm(enumerate(train_loader)):
            optimizer.zero_grad()

            images = images.to(device)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            cls_scores_batch, bbox_preds_batch, indices_batch, p3_mask_kernels, p3_mask_feats, p4_mask_kernels, p4_mask_feats, p5_mask_kernels, p5_mask_feats = model(images)

            loss = 0

            for cls_scores, pred_bboxes, indices, p3_mask_kernels, p3_mask_feats, p4_mask_kernels, p4_mask_feats, p5_mask_kernels, p5_mask_feats, target in zip(cls_scores_batch, bbox_preds_batch, indices_batch, p3_mask_kernels, p3_mask_feats, p4_mask_kernels, p4_mask_feats, p5_mask_kernels, p5_mask_feats, targets):
                mask_kernels = [p3_mask_kernels, p4_mask_kernels, p5_mask_kernels]
                mask_feats = [p3_mask_feats, p4_mask_feats, p5_mask_feats]

                labels = target['labels']
                boxes = target['boxes']
                masks = target['masks']

                matching_matrix, matching_matrix_background = label_postprocessor.calculate_cost_matrix(cls_scores, pred_bboxes, labels, boxes)
                background_preds, matched_classes, matched_pred_bboxes, matched_gt_bboxes, matched_masks, resized_gt_masks = label_postprocessor.extract_matched_pairs(matching_matrix=matching_matrix, matching_matrix_background=matching_matrix_background, cls_scores=cls_scores, pred_bboxes=pred_bboxes, indices=indices, mask_kernels=mask_kernels, mask_feats=mask_feats, gt_labels=labels, gt_bboxes=boxes, gt_masks=masks, img_size=model.img_size)

                loss += calculate_loss(background_preds, matched_classes, matched_pred_bboxes, matched_gt_bboxes, matched_masks, resized_gt_masks)

            loss.backward()
            optimizer.step()
            
            if batch_idx % PERFORMACE_OUTPUT_STEPS == 0:
                print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item()}")

        # evaluate model every epoch
        evaluate_model(model, valid_loader, device)



if __name__ == "__main__":
    logging.basicConfig(
        format='%(asctime)s %(levelname)s:%(message)s',
        level=logging.INFO,
        datefmt='%m/%d/%Y %I:%M:%S %p',
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=str, required=False, default="./data/instance_segmentation_coco")
    parser.add_argument("--batch-size", type=int, required=False, default=16)
    parser.add_argument("--epochs", type=int, required=False, default=10)
    parser.add_argument("--lr", type=float, required=False, default=0.0001)
    parser.add_argument("--device", type=str, required=False, default="cpu")
    parser.add_argument("--img-size", type=int, required=False, default=1024)
    parser.add_argument("--save-dir", type=str, required=False, default="./")
    parser.add_argument("--model-name", type=str, required=False, default="rtmdet_model")

    args = parser.parse_args()


    logging.info(f"Loading dataset from {args.dataset_dir}")
    train_loader, valid_loader, test_loader, num_classes = get_coco_dataloaders(args.dataset_dir, args.batch_size, img_size=args.img_size) 

    logging.info(f"Loading model")
    model = RTMDETModel(num_classes=num_classes, img_size=args.img_size, device=args.device)

    logging.info(f"Begin training model")
    train_rtmdet_model(model, train_loader, valid_loader, test_loader, args.epochs, args.lr, args.device)

    logging.info(f"Train complete. Saving model to {args.save_dir}/{args.model_name}")
    model.save_pretrained(args.save_dir, args.model_name)