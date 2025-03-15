"""Split YOLO with images exported files into train/valid set

File structure after exporting `YOLO with Images` format from label-studio:

- images
- labels
- classes.txt
- notes.json

We need to
1) Split images and labels to train/valid 
2) Create metadata data.yaml file using classes.txt


# Output file structure:
- train
   - images
   - labels
- valid
   - images
   - labels
- data.yaml

# data.yaml example:
```yaml
train: ./train/images
valid: ./valid/images

nc: 4
names: ['sprite_idle_down', 'sprite_idle_up', 'sprite_idle_right', 'sprite_idle_left']
```

# Run Example: In nino base directory,
```sh
python -m scripts.split_yolo_exported_files --input-dir ./data/instance_segmentation_yolo --output-dir ./data/instance_segmentation_yolo
```
"""

import argparse
import os
import shutil
import logging

from pathlib import Path
from tqdm import tqdm
from sklearn.model_selection import train_test_split

CLASS_TEXT_FILE_NAME = 'classes.txt'
METADATA_FILE_NAME = 'data.yaml'

def create_data_yaml_metadata_file(*, input_dir, output_dir):
    class_text_file_dir = Path(input_dir) / CLASS_TEXT_FILE_NAME 
    metadata_file_path = Path(output_dir) / METADATA_FILE_NAME
    
    try:
        with open(class_text_file_dir, 'r') as f:
            classes = [line.rstrip() for line in f]
    except FileNotFoundError:
        raise FileNotFoundError(f'file not found: {class_text_file_dir}')

    with open(metadata_file_path, 'w') as f:
        f.write('train: ./train/images\n')
        f.write('valid: ./valid/images\n\n')
        f.write(f'nc: {len(classes)} \n')
        f.write(f'names: {classes}')

def split_images_labels_to_train_valid(*, train_ratio_percent, input_dir, output_dir): 
    """Split total dataset into train/valid using train_ratio_percent value
    
    * train_ratio_percent: integer between 1 and 99. Defines train set's ratio.
    ex) if train_ratio_percent = 85, train/valid split is done 0.85/0.15 

    * input dir must have images/ and labels/ directory
    
    output split directory:
    - train
        - images
        - labels
    - valid
        - images
        - labels
    """
    assert train_ratio_percent > 0 and args.train_ratio_percent < 100, f"train_ratio_percent {args.train_ratio_percent} not between 1-99."
    train_ratio = train_ratio_percent / 100
    valid_ratio = (100 - train_ratio_percent) / 100
    
    logging.info(f"train valid split into {train_ratio}:{valid_ratio}")

    total_images_dir = Path(input_dir) / 'images'
    total_labels_dir = Path(input_dir) / 'labels'
    
    total_image_files = [file for file in total_images_dir.glob('**/*') if file.is_file() and (file.name.endswith('.jpg') or file.name.endswith('.png'))]
    assert len(total_image_files), f"There is no images in input dir {input_dir}"

    train_valid_image_files = train_test_split(total_image_files, train_size=train_ratio)
    train_valid_dir = [Path(input_dir) / 'train', Path(input_dir) / 'valid']


    for split_base_dir, split_image_files in tqdm(zip(train_valid_dir, train_valid_image_files)):
        images_dir = split_base_dir / 'images'
        labels_dir = split_base_dir / 'labels'

        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)

        for split_image_file in split_image_files:
            image_file_name = split_image_file.name
            label_file_name = f"{split_image_file.stem}.txt"

            shutil.copy(total_images_dir / image_file_name, images_dir / image_file_name)
            shutil.copy(total_labels_dir / label_file_name, labels_dir / label_file_name)
        

if __name__ == '__main__':
    logging.basicConfig(
        format='%(asctime)s %(levelname)s:%(message)s',
        level=logging.INFO,
        datefmt='%m/%d/%Y %I:%M:%S %p',
    )

    parser = argparse.ArgumentParser()

    parser.add_argument('--train-ratio-percent', type=int, required=False, default='80', help='percent ratio of train split. If 80, train/valid is split into 0.8/0.2. Ratio must be between 1 and 99')
    parser.add_argument('--input-dir', type=str, required=False, default='./data/instance_segmentation_yolo', help='Project ID of the Label Studio project we want to extract image from')
    parser.add_argument('--output-dir', type=str, required=False, default='./data/instance_segmentation_yolo', help='Output directory for the splitted dataset')

    args = parser.parse_args()

    train_ratio_percent = args.train_ratio_percent
    input_dir = args.input_dir
    output_dir = args.output_dir
    
    logging.info('creating data.yaml file')
    create_data_yaml_metadata_file(input_dir=input_dir, output_dir=output_dir)

    logging.info('splitting images to train/valid')
    split_images_labels_to_train_valid(train_ratio_percent=train_ratio_percent, input_dir=input_dir, output_dir=output_dir)