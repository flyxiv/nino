"""Collect sprites from gameplay footage video or a single image

Usage ex)
```sh
python scripts/collect_sprites.py --model ./trained_models/nino_seg_yolo_300e.pt --input-path ./demo/test.mp4 --output-dir ./output_data/sprites --model-type yolo --conf 30
python scripts/collect_sprites.py --model ./trained_models/nino_seg_yolo_300e.pt --input-path ./demo/test_img.png --output-dir ./output_data/sprites --model-type yolo --conf 30
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
from PIL import Image
from tqdm import tqdm
from typing import List
from util.get_segmented_sprite import collect_sprites_from_images
from util.sprite_metadata import SPRITE_NO_DIRECTION_IDS, SPRITE_NO_DIRECTIONS
from model.sprite_classification.consts import PREPROCESS_TRANSFORMS
from pathlib import Path
from util.image_video_classification import is_video, is_image

BATCH_SIZE = 8

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", type=str, required=True, help="pretrained .pt file")
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
        default="",
        help="config file for maskrcnn",
    )
    parser.add_argument(
        "--output-dir", type=str, required=False, default="./", help="output directory"
    )
    parser.add_argument(
        "--model-type",
        type=str,
        required=False,
        default="yolo",
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

    return parser.parse_args()


def parse_video_to_frames(video: str) -> List[np.ndarray]:
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
    video: str, output_dir: str, model, model_type: str, conf: float
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
            batch_frames, output_dir, model, model_type, batch_idx, conf=conf
        )


def load_model(model: str, model_type: str, config_file: str):
    if model_type == "yolo":
        from ultralytics import YOLO

        return YOLO(model)
    elif model_type == "maskrcnn":
        from mmdet.apis import init_detector

        return init_detector(config_file, model, device="cuda:0")


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s %(levelname)s:%(message)s",
        level=logging.INFO,
        datefmt="%m/%d/%Y %I:%M:%S %p",
    )

    args = parse_args()
    model = args.model
    input_path = args.input_path
    output_dir = args.output_dir
    model_type = args.model_type
    conf = args.conf / 100
    classification_model_path = args.classification_model

    model = load_model(model, model_type, args.config_file)
    classification_model = torchvision.models.efficientnet_v2_l(weights=None)
    classification_model.classifier[-1] = nn.Linear(
        in_features=classification_model.classifier[-1].in_features,
        out_features=len(SPRITE_NO_DIRECTION_IDS),
    )
    classification_model.load_state_dict(torch.load(classification_model_path))
    classification_model.eval()

    logging.info(f"Collecting sprites from {input_path}")

    input_path_str = input_path
    output_dir = Path(output_dir)
    input_path = Path(input_path)

    if is_video(input_path):
        collect_sprites_from_video(input_path, output_dir, model, model_type, conf)
    elif os.path.isdir(input_path_str):
        for img in os.listdir(input_path_str):
            if is_image(img):
                original_image = cv2.imread(os.path.join(input_path_str, img))
                original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
                collect_sprites_from_images(
                    original_image, output_dir, model, model_type, conf=conf
                )

            if is_video(img):
                collect_sprites_from_video(
                    input_path / img, output_dir, model, model_type, conf=conf
                )
    else:
        original_image = cv2.imread(input_path_str)
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

        collect_sprites_from_images(original_image, output_dir, model, model_type, conf=conf)

    logging.info(f"Classifying sprites from {output_dir}")

    for sprite_name in SPRITE_NO_DIRECTIONS.values():
        sprite_dir = output_dir / sprite_name

        if not os.path.exists(sprite_dir):
            os.makedirs(sprite_dir)

    for img_path in tqdm(os.listdir(output_dir), desc="Classifying sprites"):
        if is_image(img_path):
            img = Image.open(output_dir / img_path)
            img = PREPROCESS_TRANSFORMS(img).unsqueeze(0)
            img_classes = classification_model(img)

            highest_prob_idx = torch.argmax(img_classes, dim=1).to("cpu").item()
            sprite_name = SPRITE_NO_DIRECTIONS[highest_prob_idx]
            sprite_dir = output_dir / sprite_name

            shutil.move(output_dir / img_path, sprite_dir / img_path)
