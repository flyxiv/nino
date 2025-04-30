"""Read ImageNet Files and convert them into classification dataset

Imagenet dataset file structure:
ILSVRC
    - Annotations
        - CLS-LOC
            - train
            - val
    - Data
        - CLS-LOC
            - train
            - val
            - test
        
"""

import xml.etree.ElementTree as ET
import argparse 
import torch
import os
import pandas as pd

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from pathlib import Path
from PIL import Image
from tqdm import tqdm

IMAGENET_LABEL_COUNT = 1000

class ImageNetDataset(Dataset):
    def __init__(self, image_dir, label_dir, label_table, transform=None, train=False, img_size=1024):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.img_size = img_size
        self.label_table = label_table
        self.transform = transform

        self.dataset_info = []

        if train:
            for image_code in tqdm(self.image_dir.iterdir(), desc="Loading training dataset"):
                for image_path in image_code.iterdir():
                    label_path = self.label_dir / image_code.name / f"{image_path.stem}.xml"

                    self.dataset_info.append({'image': image_path, 'label': label_path})
        else:
            for image_path in tqdm(self.image_dir.iterdir(), desc="Loading valid dataset"):
                label_path = self.label_dir / f"{image_path.stem}.xml"

                self.dataset_info.append({'image': image_path, 'label': label_path})

    def __len__(self):
        return len(self.dataset_info)

    def __getitem__(self, idx):
        image_path = self.dataset_info[idx]['image']
        label_path = self.dataset_info[idx]['label']

        img = Image.open(image_path)
        img = img.resize((self.img_size, self.img_size))

        if self.transform:
            img = self.transform(img)
        else:
            img = transforms.ToTensor()(img)

        target = self._parse_label_xml_file(label_path)

        return img, target

    def _parse_label_xml_file(self, label_path):
        tree = ET.parse(label_path)
        root = tree.getroot()

        folder = root.find('folder').text

        if not folder.startswith('n'):
            folder = 'n' + folder

        target = {
            'image_name': root.find('filename').text,
            'labels': self.label_table[folder],
        }

        return target

def collate_fn_images(batch):
    images = []
    targets = []

    for img, tgt in batch:
        if img.size(0) == 1:
            img = img.repeat(3, 1, 1)
        if img.size(0) > 3:
            img = img[:3, :, :]

        images.append(img)
        targets.append(tgt)

    images = torch.stack(images, dim=0)
    return images, targets


def parse_arguments():
    parser.add_argument('--label_csv_path', type=str, default=None, help='CSV file that maps label code to label id starting from 0')
    return parser.parse_args()


def clean_dataset(image_dir, label_dir, train=True):
    if train:
        for image_code in tqdm(image_dir.iterdir(), desc="Loading training dataset"):
            for image_path in image_code.iterdir():
                label_path = label_dir / image_code.name / f"{image_path.stem}.xml"

#                if not label_path.exists():
#                    os.remove(image_path)

                img = Image.open(image_path)
                if img.mode == 'L':
                    print(img.mode)
                    print(f"Removing {image_path} because it is a grayscale image")
                    os.remove(image_path)

    else:
        for image_path in tqdm(image_dir.iterdir(), desc="Loading valid dataset"):
            label_path = label_dir / f"{image_path.stem}.xml"

#            if not label_path.exists():
#                os.remove(image_path)
            img = Image.open(image_path)
            if img.mode == 'L':
                print(f"Removing {image_path} because it is a grayscale image")
                os.remove(image_path)



def get_imagenet_dataloaders(imagenet_path, label_csv_path, batch_size, img_size):
    """ Read all images from the ImageNet dataset
    Args:
        imagenet_path (str): Path to ImageNet dataset root directory ex) imagenet-object-localization-challenge/ILSVRC
        label_csv_path (str): CSV file with following columns:
            label_code: label code (ex) n01440764)
            label_id: label id starting from 0
        img_size (int): Image size in pixels
    Returns:
        Train/Valid dataloaders 
    """
    label_pd_table = pd.read_csv(label_csv_path)
    label_table = {row.label_code: row.label_id for row in label_pd_table.itertuples()}

    imagenet_image_path = Path(imagenet_path) / 'Data' / 'CLS-LOC'
    imagenet_label_path = Path(imagenet_path) / 'Annotations' / 'CLS-LOC'

    train_images = []
    val_images = []

    train_image_path = imagenet_image_path / 'train'
    val_image_path = imagenet_image_path / 'val'

    train_label_path = imagenet_label_path / 'train'
    val_label_path = imagenet_label_path / 'val'

    # TODO: Add normalization 
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_dataset = ImageNetDataset(train_image_path, train_label_path, label_table, transform=transform, img_size=img_size, train=True)
    val_dataset = ImageNetDataset(val_image_path, val_label_path, label_table, transform=transform, img_size=img_size, train=False)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn_images)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn_images)

    return train_dataloader, val_dataloader


def create_label_csv(imagenet_path, label_csv_path):
    """ Create a CSV file that maps label code to label id starting from 0
    Args:
        imagenet_path (str): Path to ImageNet dataset root directory

    Returns:
        csv file with the following columns:
            label_code: label code (ex) n01440764)
            label_id: label id starting from 0
    """
    imagenet_train_image_path = Path(imagenet_path) / 'Data' / 'CLS-LOC' / 'train'

    label_table = {'label_code': [], 'label_id': []}

    idx = 0

    for label_name in imagenet_train_image_path.iterdir():
        if label_name.is_dir():
            label_table['label_code'].append(label_name.name)
            label_table['label_id'].append(idx)
            idx += 1
    
    assert IMAGENET_LABEL_COUNT == len(label_table['label_code']), f"label_table length is {len(label_table['label_code'])}, but expected {IMAGENET_LABEL_COUNT}"

    label_pd_table = pd.DataFrame(label_table)
    label_pd_table.to_csv(label_csv_path, index=False)
