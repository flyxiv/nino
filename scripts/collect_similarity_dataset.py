"""Iterate through different sprite files and sample 500 image pairs to create a similarity dataset.

Usage ex)
```sh
python scripts/collect_similarity_dataset.py --sprite-base-dir ./output_data/sprites --output-dir ./output_data/similarity_dataset --num-samples 200 --start-index 0
```
"""

import argparse
import os
import random
import shutil

from pathlib import Path
from tqdm import tqdm
from util.image_video_classification import is_image
from util.sprite_metadata import VALID_SPRITE_DIRECTORY_NAMES, SPRITE_SIZE
from PIL import Image

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sprite-base-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--num-samples", type=int, default=200)
    parser.add_argument("--start-index", type=int, default=0)

    return parser.parse_args()

    
def collect_sprite_files(sprite_base_dir, output_dir, num_samples, start_index):
    os.makedirs(output_dir, exist_ok=True)

    sprite_dirs = [Path(sprite_base_dir) / sprite_dir for sprite_dir in os.listdir(sprite_base_dir) if os.path.isdir(os.path.join(sprite_base_dir, sprite_dir)) and sprite_dir in VALID_SPRITE_DIRECTORY_NAMES]

    for sprite_dir in sprite_dirs:
        sprite_files = [image_file for image_file in os.listdir(sprite_dir) if is_image(image_file)]

        num_samples = min(num_samples, int(len(sprite_files) ** 2 / 3))
        
        for _ in range(num_samples):
            sprite_file_1, sprite_file_2 = random.sample(sprite_files, 2)

            image_1 = Image.open(Path(sprite_dir) / sprite_file_1)
            image_2 = Image.open(Path(sprite_dir) / sprite_file_2)

            image_1 = image_1.resize(SPRITE_SIZE)
            image_2 = image_2.resize(SPRITE_SIZE)

            sprite_file_1_path = Path(output_dir) / f"{start_index}_1.png"
            sprite_file_2_path = Path(output_dir) / f"{start_index}_2.png"

            image_1.save(sprite_file_1_path)
            image_2.save(sprite_file_2_path)

            start_index += 1


if __name__ == "__main__":
    args = parse_args()
    sprite_base_dir = args.sprite_base_dir
    output_dir = args.output_dir
    num_samples = args.num_samples
    start_index = args.start_index
    
    collect_sprite_files(sprite_base_dir, output_dir, num_samples, start_index)
