"""Collect sprites from gameplay footage video or a single image

Usage ex)
```sh
python scripts/collect_sprites.py --model ./nino_seg_yolo_300e.pt --input_path ./test.mp4 --output-dir ./output_data/sprites --model-type yolo
python scripts/collect_sprites.py --model ./nino_seg_yolo_300e.pt --input_path ./test_img.png --output-dir ./output_data/sprites --model-type yolo
```
"""

import os
import cv2
import argparse
import numpy as np
from PIL import Image
import logging
from tqdm import tqdm
from typing import List
from util.get_segmented_sprite import collect_sprites_from_images
from ultralytics import YOLO
from mmdet.apis import init_detector, inference_detector

BATCH_SIZE = 8 

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

def collect_sprites_from_video(video: str, output_dir: str, model, model_type: str, conf: float):
    frames = parse_video_to_frames(video)

    batch_cnt = (len(frames) // BATCH_SIZE) + 1
    for batch_idx in range(batch_cnt):
        batch_start_idx = batch_idx * BATCH_SIZE
        batch_end_idx = min(batch_start_idx + BATCH_SIZE, len(frames))
        batch_frames = frames[batch_start_idx:batch_end_idx]

        collect_sprites_from_images(batch_frames, output_dir, model, model_type, batch_idx, conf)

def is_video(input_path: str) -> bool:
    return input_path.endswith(('.mp4', '.avi', '.mov', '.mkv'))

def load_model(model: str, model_type: str):
    if model_type == 'yolo':
        return YOLO(model, device='cuda:0')
    elif model_type == 'maskrcnn':
        return init_detector(model, device='cuda:0')

if __name__ == "__main__":
    logging.basicConfig(
        format='%(asctime)s %(levelname)s:%(message)s',
        level=logging.INFO,
        datefmt='%m/%d/%Y %I:%M:%S %p',
    )

    parser = argparse.ArgumentParser()

    parser.add_argument('--model', type=str, required=True, help='pretrained .pt file')
    parser.add_argument('--input_path', type=str, required=True, help='input image or video to get sprites from')
    parser.add_argument('--output-dir', type=str, required=False, default='./', help='output directory')
    parser.add_argument('--model-type', type=str, required=False, default='yolo', help='segmentation model type, one of [yolo, maskrcnn]')
    parser.add_argument('--conf', type=int, required=False, default=90, help='confidence threshold to save sprite')

    args = parser.parse_args()
    model = args.model
    input_path = args.input_path
    output_dir = args.output_dir
    model_type = args.model_type
    conf = args.conf / 100

    model = load_model(model, model_type)

    if is_video(input_path):
        collect_sprites_from_video(input_path, output_dir, model, model_type, conf)
    else:
        original_image = cv2.imread(input_path)
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

        collect_sprites_from_images(original_image, output_dir, model, model_type, conf)
    