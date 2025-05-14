import argparse
import torch
from torch.utils.data import Dataset
from model.duplicate_detection.consts import PREPROCESS_TRANSFORMS
from PIL import Image
from pathlib import Path
from torchvision import transforms
import shutil

import os
from sklearn.model_selection import train_test_split

MAX_IMAGE_IDX = 20000

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./output_data/labels/similar_images')
    parser.add_argument('--label-dir', type=str, default='./output_data/labels/similar_images_label')
    parser.add_argument('--split-output-dir', type=str, default='./output_data/labels/duplicate_detection')
    return parser.parse_args()

class DuplicateDetectionData:
    def __init__(self, image_path: Path, label_path: str, data_id: int):
        self.image1_path = image_path / f"{data_id}_1.png" 
        self.image2_path = image_path / f"{data_id}_2.png"
        self.label_path = label_path / f"{data_id}.txt"

        with open(self.label_path, 'r') as f:
            labels = f.read().split()
            assert len(labels) == 3, f"file {self.label_path} has {len(labels)} labels"

            clarity1, clarity2, similarity = labels

            self.image1_clarity = int(clarity1) / 100
            self.image2_clarity = int(clarity2) / 100
            self.similarity = float(similarity)

        self.data_id = data_id

class DuplicateDetectionDataTable():
    def __init__(self, data_dir: str, label_dir: str, transforms: transforms.Compose):
        self.duplicate_dataset = list()
        self.transforms = transforms

        image_base_dir = Path(data_dir)
        label_base_dir = Path(label_dir)

        for data_id in range(MAX_IMAGE_IDX):
            image1_path = image_base_dir / f"{data_id}_1.png"
            image2_path = image_base_dir / f"{data_id}_2.png"
            label_path = label_base_dir / f"{data_id}.txt"

            if not image1_path.exists() or not image2_path.exists() or not label_path.exists():
                continue

            self.duplicate_dataset.append(DuplicateDetectionData(image_base_dir, label_base_dir, data_id))
    
    def split_to_train_val_test(self, train_ratio: float = 0.8, val_ratio: float = 0.1, test_ratio: float = 0.1):
        train_data, val_test_data = train_test_split(self.duplicate_dataset, test_size=test_ratio, random_state=42)
        val_data, test_data = train_test_split(val_test_data, test_size=val_ratio / (val_ratio + test_ratio), random_state=42)

        self.train_dataset = DuplicateDetectionDataset(train_data, self.transforms)
        self.val_dataset = DuplicateDetectionDataset(val_data, self.transforms)
        self.test_dataset = DuplicateDetectionDataset(test_data, self.transforms)

    def save(self, input_dir: str, output_base_dir: str):
        splits = [('train', self.train_dataset), ('val', self.val_dataset), ('test', self.test_dataset)]

        for split, dataset in splits:
            output_dir = Path(output_base_dir) / split

            if not os.path.exists(output_dir):
                os.makedirs(output_dir / 'images', exist_ok=True)
                os.makedirs(output_dir / 'labels', exist_ok=True)

            for data in dataset.data:
                image1_path, image2_path, label_path = data.image1_path, data.image2_path, data.label_path

                shutil.copy(image1_path, Path(output_dir) / 'images' / image1_path.name)
                shutil.copy(image2_path, Path(output_dir) / 'images' / image2_path.name)
                shutil.copy(label_path, Path(output_dir) / 'labels' / label_path.name)


class DuplicateDetectionDataset(Dataset):
    def __init__(self, data: list[DuplicateDetectionData], transforms: transforms.Compose):
        self.data = data
        self.transforms = transforms

    @staticmethod
    def create_from_dir(data_base_dir: Path, transforms: transforms.Compose):
        data = list()

        split = ['train', 'val', 'test']

        dataset = list()

        for split in split:
            label_dir = data_base_dir / split / 'labels'
            image_dir = data_base_dir / split / 'images'

            for label_file in label_dir.glob('*.txt'):
                data_id = label_file.stem
                data.append(DuplicateDetectionData(image_dir, label_dir, data_id))

            dataset.append(DuplicateDetectionDataset(data, transforms))

        return dataset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image1_path, image2_path = self.data[index].image1_path, self.data[index].image2_path

        label1_clarity = self.data[index].image1_clarity
        label2_clarity = self.data[index].image2_clarity
        label_similarity = self.data[index].similarity

        image1 = Image.open(image1_path)
        image2 = Image.open(image2_path)

        return self.transforms(image1), self.transforms(image2), torch.tensor(label1_clarity), torch.tensor(label2_clarity), torch.tensor(label_similarity)

    def save(self, input_dir: str, output_dir: str):
        output_dir = os.path.join(output_dir, self.split)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        for data in self.data:
            image_path, label = data.image_path, data.label
            image = Image.open(image_path)

            image.save(os.path.join(output_dir, f'{label}.png'))

def collate_duplicate_detection_fn(batch):
    images1, images2, label1_clarity, label2_clarity, label_similarity = zip(*batch)

    return torch.stack(images1), torch.stack(images2), torch.reshape(torch.stack(label1_clarity, dim=0), [-1, 1]), torch.reshape(torch.stack(label2_clarity, dim=0), [-1, 1]), torch.reshape(torch.stack(label_similarity, dim=0), [-1, 1])

if __name__ == '__main__':
    args = parse_args()

    split_output_dir = args.split_output_dir
    data_dir = args.data_dir
    label_dir = args.label_dir

    dataset = DuplicateDetectionDataTable(data_dir, label_dir, PREPROCESS_TRANSFORMS)
    dataset.split_to_train_val_test()
    dataset.save(data_dir, split_output_dir)