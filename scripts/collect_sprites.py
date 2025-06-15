"""Collect sprites from gameplay footage video or a single image

Usage ex)
```sh
python scripts/collect_sprites.py --model ./trained_models/nino_seg_yolo_300e.pt --input-path ./demo/test.mp4 --output-dir ./output_data/sprites --model-type yolo --conf 30
python scripts/collect_sprites.py --model ./trained_models/nino_seg_yolo_300e.pt --input-path ./demo/test_img.png --output-dir ./demo_out --model-type yolo --conf 30

python -m scripts.collect_sprites --input-path ./input --output-dir ./output_data --segmentation-model-type maskrcnn --classification-model ./trained_models/nino_classification_efficientnet_v2_l.pth --duplicate-detection-model ./trained_models/nino_duplicate_detection.pth
```
"""

import os
import cv2
import argparse
import shutil
import logging
import torch
import torch.nn as nn
import torchvision.models
import numpy as np
from datetime import datetime
from PIL import Image
from tqdm import tqdm
from typing import List
from util.get_segmented_sprite import collect_sprites_from_images
from util.sprite_metadata import SPRITE_NO_DIRECTION_IDS, SPRITE_NO_DIRECTIONS
from model.sprite_classification.consts import PREPROCESS_TRANSFORMS
from model.duplicate_detection.efficientnet.duplicate_detection_model import DuplicateDetectionModel
from pathlib import Path
from util.image_video_classification import is_video, is_image

BATCH_SIZE = 8

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input-path",
        type=str,
        required=True,
        help="input image or video to get sprites from",
    )
    parser.add_argument(
        "--config-file",
        type=str,
        required=False,
        default="./model/segmentation/mask_rcnn_resnet50/resnet_config.py",
        help="config file for maskrcnn",
    )
    parser.add_argument(
        "--output-dir", type=str, required=False, default="./", help="output directory"
    )
    parser.add_argument(
        "--segmentation-model-type",
        type=str,
        choices=["yolo", "maskrcnn"],
        required=False,
        default="maskrcnn",
        help="segmentation model type, one of [yolo, maskrcnn]",
    )
    parser.add_argument(
        "--conf",
        type=int,
        required=False,
        default=90,
        help="confidence threshold to save sprite",
    )
    parser.add_argument(
        "--classification-model",
        type=str,
        required=False,
        default="./trained_models/nino_classification_efficientnet_v2_l.pth",
        help="path for classification model",
    )
    parser.add_argument(
        "--duplicate-detection-model",
        type=str,
        required=False,
        default="./trained_models/nino_duplicate_detection.pth",
    )

    return parser.parse_args()


def parse_video_to_frames(video: Path) -> List[np.ndarray]:
    cap = cv2.VideoCapture(video)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    cap.release()
    return frames


def collect_sprites_from_video(
    video: Path, output_dir: Path, model, segmentation_model_type: str, conf: float
):
    frames = parse_video_to_frames(video)

    batch_cnt = (len(frames) // BATCH_SIZE) + 1
    for batch_idx in tqdm(range(batch_cnt), desc="Collecting sprites from video"):
        batch_start_idx = batch_idx * BATCH_SIZE
        batch_end_idx = min(batch_start_idx + BATCH_SIZE, len(frames))
        batch_frames = frames[batch_start_idx:batch_end_idx]

        if len(batch_frames) == 0:
            break

        collect_sprites_from_images(
            batch_frames, output_dir, model, segmentation_model_type, batch_idx, conf=conf
        )


def load_model(segmentation_model_type: str, config_file: str):
    if segmentation_model_type == "yolo":
        from ultralytics import YOLO

        model = './trained_models/nino_seg_yolo_300e.pt'
        return YOLO(model)
    elif segmentation_model_type == "maskrcnn":
        from mmdet.apis import init_detector

        model = './trained_models/nino_seg_maskrcnn_e500.pth'
        return init_detector(config_file, model, device="cuda:0")


def detect_sprites_from_input(input_path: Path, output_dir: Path, segmentation_model_type: str, conf: float):
    model = load_model(segmentation_model_type, args.config_file)

    if is_video(input_path):
        collect_sprites_from_video(input_path, output_dir, model, segmentation_model_type, conf)
    elif is_image(input_path.name):
        original_image = cv2.imread(str(input_path))
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

        collect_sprites_from_images(original_image, output_dir, model, segmentation_model_type, conf=conf)
    elif input_path.is_dir():
        for img in input_path.iterdir():
            if is_image(img.name):
                original_image = cv2.imread(str(img))
                original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
                collect_sprites_from_images(
                    original_image, output_dir, model, segmentation_model_type, conf=conf
                )

            if is_video(img.name):
                collect_sprites_from_video(
                    img, output_dir, model, segmentation_model_type, conf=conf
                )
    else:
        raise ValueError(f"Invalid input path: {input_path}. Please provide a valid video, image, or directory path.")


def classify_sprites(output_dir: Path, classification_model_path: str, device: str):
    classification_model = torchvision.models.efficientnet_v2_l(weights=None)
    classification_model.classifier[-1] = nn.Linear(
        in_features=classification_model.classifier[-1].in_features,
        out_features=len(SPRITE_NO_DIRECTION_IDS),
    )
    classification_model.load_state_dict(torch.load(classification_model_path, map_location=device))
    classification_model = classification_model.to(device)
    classification_model.eval()

    for sprite_name in SPRITE_NO_DIRECTIONS.values():
        sprite_dir = output_dir / sprite_name

        if not os.path.exists(sprite_dir):
            os.makedirs(sprite_dir)

    for img_path in tqdm(os.listdir(output_dir), desc="Classifying sprites"):
        if is_image(img_path):
            img = Image.open(output_dir / img_path)
            img = PREPROCESS_TRANSFORMS(img).unsqueeze(0).to(device)
            img_classes = classification_model(img)

            highest_prob_idx = torch.argmax(img_classes, dim=1).to("cpu").item()
            sprite_name = SPRITE_NO_DIRECTIONS[highest_prob_idx]
            sprite_dir = output_dir / sprite_name

            shutil.move(output_dir / img_path, sprite_dir / img_path)

def remove_duplicate_sprites(output_dir: Path, duplicate_detection_model_path: str, device: str):
    model = DuplicateDetectionModel()
    model.load_state_dict(torch.load(duplicate_detection_model_path))
    model = model.to(device)
    model.eval()

    for sprite_name in SPRITE_NO_DIRECTIONS.values():
        sprite_dir = output_dir / sprite_name
        
        sprite_images = list(sprite_dir.glob("*.png"))
        valid_sprites = {img.name: True for img in sprite_images}

        for i in tqdm(range(len(sprite_images))):
            img1_name = sprite_images[i].name
            if not valid_sprites[img1_name]:
                continue

            img1 = Image.open(sprite_dir / img1_name)
            img1_tensor = PREPROCESS_TRANSFORMS(img1).unsqueeze(0).to(device)

            for j in range(i + 1, len(sprite_images)):
                img2_name = sprite_images[j].name
                if not valid_sprites[img2_name]:
                    continue

                img2 = Image.open(sprite_dir / img2_name)
                img2_tensor = PREPROCESS_TRANSFORMS(img2).unsqueeze(0).to(device)

                img1_clarity, img2_clarity, similarity = model(img1_tensor, img2_tensor)

                if max(img1_clarity, img2_clarity) < 0.8:
                    valid_sprites[img1_name] = False
                    valid_sprites[img2_name] = False
                    break

                if similarity > 0.9:
                    if img1_clarity > img2_clarity:
                        valid_sprites[img2_name] = False
                    else:
                        valid_sprites[img1_name] = False
                        break

        for img in sprite_images:
            if not valid_sprites[img.name]:
                logging.info(f"removing duplicate sprite {img}") 
                os.remove(img)


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s %(levelname)s:%(message)s",
        level=logging.INFO,
        datefmt="%m/%d/%Y %I:%M:%S %p",
    )

    task_id = datetime.now().strftime("%Y%m%d%H%M%S%f")

    args = parse_args()
    input_path = args.input_path
    output_dir = f"{args.output_dir}/{task_id}"
    segmentation_model_type = args.segmentation_model_type
    conf = args.conf / 100
    classification_model_path = args.classification_model
    duplicate_detection_model_path = args.duplicate_detection_model

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    os.makedirs(output_dir, exist_ok=True)

    input_path = Path(input_path)
    output_dir = Path(output_dir)

    logging.info(f"Collecting sprites from {input_path}")
    detect_sprites_from_input(input_path, output_dir, segmentation_model_type, conf)

    logging.info(f"Classifying sprites from {output_dir}")
    classify_sprites(output_dir, classification_model_path, device)

    logging.info(f"Removing duplicate sprites")
    remove_duplicate_sprites(output_dir, duplicate_detection_model_path, device)
