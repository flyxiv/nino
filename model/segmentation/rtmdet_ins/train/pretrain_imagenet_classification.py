"""Script for training RTMDET model backbone for classification task with ImageNet

run ex) in nino base directory,
```sh
python -m model.segmentation.train.pretrain_imagenet_classification --dataset-dir ./data/imagenet-object-localization-challenge/ILSVRC --batch-size 16 --epochs 10 --lr 0.0001 --device cuda --img-size 1024 --save-dir ./ --model-name rtmdet_backbone_classifier
```
"""

import argparse 
import logging
import torch
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss

from pathlib import Path
from tqdm import tqdm
from torch.optim import AdamW

from preprocessing.imagenet_dataset import get_imagenet_dataloaders, create_label_csv
from model.segmentation.loss.loss_functions import calculate_classification_loss
from model.segmentation.rtmdet_backbone_classifier import RTMDetBackboneClassifier
from accelerate import Accelerator

PERFORMACE_OUTPUT_STEPS = 100 

def evaluate_model(model, valid_loader, epoch, device):
    """Evaluate RTMDET backbone classifier
    """

    model.eval()

    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(valid_loader):
            cls_scores_batch = model(images)
            labels = torch.tensor([label['labels'] for label in target], device=device)
            loss = calculate_classification_loss(cls_scores_batch, labels)
            print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item()}")


def calculate_classification_loss(logits, targets, label_smoothing=0.1):
    """Calculate classification loss with label smoothing
    Args:
        logits: Model output logits
        targets: Ground truth labels
        label_smoothing: Label smoothing factor
    """
    num_classes = logits.size(-1)
    # Apply log_softmax to get log probabilities
    log_probs = F.log_softmax(logits, dim=-1)
    
    # Create smoothed labels
    with torch.no_grad():
        true_dist = torch.zeros_like(logits)
        true_dist.fill_(label_smoothing / (num_classes - 1))
        true_dist.scatter_(1, targets.unsqueeze(1), 1.0 - label_smoothing)
    
    loss = -(true_dist * log_probs).sum(dim=-1).mean()
    return loss

def pretrain_rtmdet_backbone(model, train_loader, valid_loader, epochs, lr, device):
    """Pre-train RTMDET backbone classifier
    """
    accelerator = Accelerator()
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)  # Added weight decay
    device = accelerator.device

    if device == 'cuda':
        model = model.to(device)

    model, optimizer, train_loader = accelerator.prepare(model, optimizer, train_loader)

    for epoch in range(epochs):
        for batch_idx, (image, target) in tqdm(enumerate(train_loader)):
            optimizer.zero_grad()

            images = image.to(device)
            cls_scores_batch = model(images)

            labels = torch.tensor([label['labels'] for label in target], device=device)
            loss = calculate_classification_loss(cls_scores_batch, labels)

            accelerator.backward(loss)
            optimizer.step()
            
            if batch_idx % PERFORMACE_OUTPUT_STEPS == 0:
                with torch.no_grad():
                    print(f"Epoch {epoch}, Batch {batch_idx}")
                    print(f"Loss: {loss.item():.4f}")
                    print(f"Predicted class: {predictions[0]}, True class: {labels[0]}")
                    print(f"Number of unique predictions in batch: {predictions.unique().size(0)}")

        # evaluate model every epoch
        evaluate_model(model, valid_loader, epoch, device)

    model.save_backbone(args.save_dir, args.model_name)



if __name__ == "__main__":
    logging.basicConfig(
        format='%(asctime)s %(levelname)s:%(message)s',
        level=logging.INFO,
        datefmt='%m/%d/%Y %I:%M:%S %p',
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=str, required=False, default="./data/imagenet")
    parser.add_argument("--label-csv-path", type=str, required=False, default="./data/imagenet_label_table.csv")
    parser.add_argument("--batch-size", type=int, required=False, default=16)
    parser.add_argument("--epochs", type=int, required=False, default=10)
    parser.add_argument("--lr", type=float, required=False, default=0.001)
    parser.add_argument("--device", type=str, required=False, default="cpu")
    parser.add_argument("--img-size", type=int, required=False, default=1024)
    parser.add_argument("--save-dir", type=str, required=False, default="./")
    parser.add_argument("--model-name", type=str, required=False, default="rtmdet_classifier")
    parser.add_argument("--seed", type=int, required=False, default=42)

    args = parser.parse_args()

    torch.manual_seed(args.seed)

    if not Path(args.label_csv_path).exists():
        create_label_csv(args.dataset_dir, args.label_csv_path)

    logging.info(f"Loading dataset from {args.dataset_dir}")
    train_loader, valid_loader = get_imagenet_dataloaders(args.dataset_dir, args.label_csv_path, args.batch_size, img_size=args.img_size) 

    logging.info(f"Loading model")
    model = RTMDetBackboneClassifier(img_size=args.img_size, device=args.device)

    logging.info(f"Begin pre-training.")
    pretrain_rtmdet_backbone(model, train_loader, valid_loader, args.epochs, args.lr, args.device)

    logging.info(f"Train complete. Saving model to {args.save_dir}/{args.model_name}")
    model.save_pretrained(args.save_dir, args.model_name)