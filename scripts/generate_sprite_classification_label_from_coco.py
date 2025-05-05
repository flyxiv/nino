"""Load images in the coco dataset and generate sprite classification labels.

1) Load the coco dataset
2) Crop bounding boxes from the images
3) Decode segmentation labels to masks
4) Multiply bbox with mask to get sprite
5) Get sprite classification label from annotation

input directory must be in COCO structure:

data/coco/
    annotations/
        instances_train2017.json
        instances_val2017.json
    train2017/
        *.jpg
    val2017/
        *.jpg
"""

import cv2
import os
import argparse
import numpy as np
import pycocotools.mask as mask_utils

from tqdm import tqdm
from pycocotools.coco import COCO

from util.sprite_classifications import SPRITE_TABLE, SPRITE_IDS

def parse_args():
    parser = argparse.ArgumentParser(description='Generate sprite classification labels from COCO dataset')
    parser.add_argument('--input-dir', type=str, required=True, help='Path to the COCO dataset directory')
    parser.add_argument('--output-dir', type=str, required=True, help='Path to save the generated sprite classification labels')
    return parser.parse_args()

def cut_label_prefix(label: str):
    prefix1 = "character_"
    prefix2 = "sprite_"

    if label.startswith(prefix1):
        label = label[len(prefix1):]

    if label.startswith(prefix2):
        label = label[len(prefix2):]

    return label


def main(input_dir, output_dir):
    splits = ['train', 'val', 'test']
    coco_annotations = [COCO(os.path.join(input_dir, 'annotations', f'instances_{split}2017.json')) for split in splits]
    image_dirs = [os.path.join(input_dir, split + '2017') for split in splits]

    for split, coco, image_dir in zip(splits, coco_annotations, image_dirs):
        categories = coco.loadCats(coco.getCatIds())

        for img_id in tqdm(coco.getImgIds()):
            img_info = coco.loadImgs(img_id)[0]
            img = cv2.imread(os.path.join(image_dir, img_info['file_name']))

            # Get annotation
            ann_ids = coco.getAnnIds(img_id)
            anns = coco.loadAnns(ann_ids)

            for ann_idx, ann in enumerate(anns):
                # Get sprite classification label
                sprite_classification_label = SPRITE_TABLE[ann['category_id']]

                # Extract sprites
                rles = mask_utils.frPyObjects(ann['segmentation'], img_info['height'], img_info['width'])

                mask = mask_utils.decode(rles)
                mask = cv2.resize(mask, (img.shape[1], img.shape[0]))

                # 마스크 적용 (마스크=1: 원본 이미지, 마스크=0: 흰색 배경)
                white_background = np.ones_like(img) * 255
                masked_crop = np.zeros_like(img)
                
                for c in range(3):  # RGB 채널
                    masked_crop[:, :, c] = np.where(mask == 1, 
                                                    img[:, :, c], 
                                                    white_background[:, :, c])

                x1, y1, w, h = ann['bbox']

                x1 = max(0, int(x1))
                y1 = max(0, int(y1))
                x2 = min(img.shape[1], int(x1 + w))
                y2 = min(img.shape[0], int(y1 + h))

                sprite = masked_crop[y1:y2, x1:x2].copy()

                img_save_path = os.path.join(output_dir, sprite_classification_label)

                if not os.path.isdir(img_save_path):
                    os.makedirs(img_save_path, exist_ok=True)

                img_name = f'{img_info["file_name"]}_{ann_idx}'

                cv2.imwrite(os.path.join(img_save_path, f'{img_name}.png'), sprite)

                with open(os.path.join(img_save_path, f'{img_name}.txt'), 'w') as f:
                    f.write(sprite_classification_label)

if __name__ == '__main__':
    args = parse_args()
    main(args.input_dir, args.output_dir)

