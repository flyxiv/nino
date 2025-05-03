"""Split COCO dataset into train/val/test splits.

Split ratio: 70:15:15

Input directory: COCO format exported from label studio.

contains:
    - images/
    - result.json

Output directory:
    - images/
    - train.json
    - valid.json
    - test.json 

run ex)
```sh
python -m preprocessing.coco_dataset_split \
    --dataset-dir ./data/instance_segmentation/coco 
```
"""

import os
import cv2
import argparse
import sklearn
import json
import logging
import shutil

from pathlib import Path
from sklearn.model_selection import train_test_split

TRAIN_RATIO = 0.85
VALID_RATIO = 0.05
TEST_RATIO = 0.10

logger = logging.getLogger(__name__)

def split_coco_dataset(dataset_dir: Path):
    annotations_dir = dataset_dir / "annotations"
    os.makedirs(annotations_dir, exist_ok=True)

    coco_data = json.load(open(dataset_dir / "result.json"))

    all_images = coco_data["images"]
    for img in all_images:
        img_name = Path(img['file_name']).name
        img['file_name'] = img_name 

    coco_data["categories"] = [{"id": 0, "name": "portrait"}, {"id": 1, "name": "character_sprite"}] 

    for ann in coco_data["annotations"]:
        ann["category_id"] = min(ann["category_id"], 1)
    
    image_ids = [img["id"] for img in all_images]
    
    train_ids, valid_test_ids = train_test_split(image_ids, train_size=TRAIN_RATIO, random_state=42)

    valid_ratio_adjusted = VALID_RATIO / (VALID_RATIO + TEST_RATIO)
    valid_ids, test_ids = train_test_split(valid_test_ids, train_size=valid_ratio_adjusted, random_state=42)
    
    train_ids_set = set(train_ids)
    valid_ids_set = set(valid_ids)
    test_ids_set = set(test_ids)
    
    train_images = []
    valid_images = []
    test_images = []
    
    for img in all_images:
        if img["id"] in train_ids_set:
            train_images.append(img)
        elif img["id"] in valid_ids_set:
            valid_images.append(img)
        elif img["id"] in test_ids_set:
            test_images.append(img)

    all_annotations = coco_data["annotations"]

    train_annotations = []
    valid_annotations = []
    test_annotations = []
    
    for ann in all_annotations:
        if ann["image_id"] in train_ids_set:
            train_annotations.append(ann)
        elif ann["image_id"] in valid_ids_set:
            valid_annotations.append(ann)
        elif ann["image_id"] in test_ids_set:
            test_annotations.append(ann)

    train_coco = {
        "images": train_images,
        "annotations": train_annotations,
        "categories": coco_data["categories"]
    }
    
    valid_coco = {
        "images": valid_images,
        "annotations": valid_annotations,
        "categories": coco_data["categories"]
    }
    
    test_coco = {
        "images": test_images,
        "annotations": test_annotations,
        "categories": coco_data["categories"]
    }
    
    with open(annotations_dir / "instances_train2017.json", "w") as f:
        json.dump(train_coco, f, indent=4)
    
    with open(annotations_dir / "instances_val2017.json", "w") as f:
        json.dump(valid_coco, f, indent=4)
    
    with open(annotations_dir / "instances_test2017.json", "w") as f:
        json.dump(test_coco, f, indent=4)

    
    
    os.makedirs(dataset_dir / "train2017", exist_ok=True)
    os.makedirs(dataset_dir / "val2017", exist_ok=True)
    os.makedirs(dataset_dir / "test2017", exist_ok=True)

    logger.info(f"Copying images to correct split.")

    for img in train_images:
        shutil.copy(dataset_dir / 'images' / img["file_name"], dataset_dir / "train2017" / img["file_name"])

    for img in valid_images:
        shutil.copy(dataset_dir / 'images' / img["file_name"], dataset_dir / "val2017" / img["file_name"])

    for img in test_images:
        shutil.copy(dataset_dir / 'images' / img["file_name"], dataset_dir / "test2017" / img["file_name"])

    
    logger.info(f"Split complete!")
    logger.info(f"Train set: {len(train_images)} images, {len(train_annotations)} annotations")
    logger.info(f"Valid set: {len(valid_images)} images, {len(valid_annotations)} annotations")
    logger.info(f"Test set: {len(test_images)} images, {len(test_annotations)} annotations")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset-dir", type=str, required=True)

    args = parser.parse_args()
    dataset_dir = Path(args.dataset_dir)

    split_coco_dataset(dataset_dir)
