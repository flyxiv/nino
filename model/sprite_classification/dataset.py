import argparse
import torch
from dataclasses import dataclass
from torch.utils.data import Dataset
from util.sprite_classifications import SPRITE_IDS, SPRITE_TABLE
from PIL import Image
from pathlib import Path
from torchvision import transforms

import os
from sklearn.model_selection import train_test_split

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./output_data/sprites')
    parser.add_argument('--split-output-dir', type=str, default='./output_data/classification')
    return parser.parse_args()

@dataclass
class SpriteClassificationData:
    image_path: str
    label: int 

class SpriteClassificationDataTable():
    def __init__(self, data_dir: str, transforms: transforms.Compose):
        self.classification_data = list()
        self.transforms = transforms

        for sprite_name in SPRITE_IDS.keys():
            for file in os.listdir(os.path.join(data_dir, sprite_name)):
                label = SPRITE_IDS[sprite_name]
                if file.endswith('.png') or file.endswith('.jpg'):
                    self.classification_data.append(SpriteClassificationData(os.path.join(data_dir, sprite_name, file), label))
    
    def split_to_train_val_test(self, train_ratio: float = 0.8, val_ratio: float = 0.1, test_ratio: float = 0.1):
        train_data, val_test_data = train_test_split(self.classification_data, test_size=test_ratio, random_state=42)
        val_data, test_data = train_test_split(val_test_data, test_size=val_ratio / (val_ratio + test_ratio), random_state=42)

        self.train_dataset = SpriteClassificationDataset(train_data, self.transforms)
        self.val_dataset = SpriteClassificationDataset(val_data, self.transforms)
        self.test_dataset = SpriteClassificationDataset(test_data, self.transforms)

    def save(self, input_dir: str, output_base_dir: str):
        splits = [('train', self.train_dataset), ('val', self.val_dataset), ('test', self.test_dataset)]

        for split, dataset in splits:
            for sprite_name in SPRITE_IDS.keys():
                output_dir = os.path.join(output_base_dir, split, sprite_name)

                if not os.path.exists(output_dir):
                    os.makedirs(output_dir, exist_ok=True)

            for data in dataset.data:
                image_path, label = data.image_path, data.label
                image_name = Path(image_path).name
                sprite_name = SPRITE_TABLE[label]

                image = Image.open(image_path)
                image.save(os.path.join(output_base_dir, split, sprite_name, image_name))


class SpriteClassificationDataset(Dataset):
    def __init__(self, data: list[SpriteClassificationData], transforms: transforms.Compose):
        self.data = data
        self.transforms = transforms

    @staticmethod
    def create_from_dir(data_base_dir: str, transforms: transforms.Compose):
        data = list()

        split = ['train', 'val', 'test']

        dataset = list()

        for split in split:
            for sprite_name in SPRITE_IDS.keys():
                for file in os.listdir(os.path.join(data_base_dir, split, sprite_name)):
                    label = SPRITE_IDS[sprite_name]

                    if file.endswith('.png') or file.endswith('.jpg'):
                            data.append(SpriteClassificationData(os.path.join(data_base_dir, split, sprite_name, file), label))

            dataset.append(SpriteClassificationDataset(data, transforms))

        return dataset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image_path, label = self.data[index].image_path, self.data[index].label

        image = Image.open(image_path)

        return self.transforms(image), label

    def save(self, input_dir: str, output_dir: str):
        output_dir = os.path.join(output_dir, self.split)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        for data in self.data:
            image_path, label = data.image_path, data.label
            image = Image.open(image_path)

            image.save(os.path.join(output_dir, f'{label}.png'))


if __name__ == '__main__':
    args = parse_args()

    split_output_dir = args.split_output_dir
    data_dir = args.data_dir

    dataset = SpriteClassificationDataTable(data_dir, transforms.Compose([transforms.ToTensor()]))
    dataset.split_to_train_val_test()
    dataset.save(data_dir, split_output_dir)
