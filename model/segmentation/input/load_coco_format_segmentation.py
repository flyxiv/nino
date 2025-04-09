"""Load COCO format segmentation data into pytorch dataloader.

1) Reads COCO json label data and image data
2) Resizes images and masks to the same size
3) Converts segment polygon to bbox and mask label
4) Load to pytorch dataloader

input: dataset directory must have:
    - images: directory of all images
    - train.json/valid.json/test.json: COCO format annotation file

"""

from pathlib import Path
import os
import json
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from pycocotools.coco import COCO
from pycocotools import mask as coco_mask
import cv2
from tqdm import tqdm

TRAIN_ANNOTATION_FILE = "train.json"
VALID_ANNOTATION_FILE = "valid.json"
TEST_ANNOTATION_FILE = "test.json"

class COCOSegmentationDataset(Dataset):
    def __init__(self, root_dir, ann_file, transform=None, img_size=1024):
        """
        COCO Segmentation 데이터셋을 로드하는 클래스

        Args:
            root_dir: base directory of the dataset
            ann_file: COCO format annotation file
            transform: callable, optional, transform to apply to the image
            img_size: size to resize the image and mask. Currently only support square size
        """

        assert img_size > 0, "img_size must be positive"    
        assert isinstance(img_size, int), "img_size must be an integer"
        
        self.root_dir = root_dir
        self.coco = COCO(Path(root_dir) / ann_file)
        self.ids = list(sorted(self.coco.imgs.keys()))
        self.transform = transform
        self.img_size = (img_size, img_size)
        
    def __len__(self):
        return len(self.ids)
    
    def __getitem__(self, idx):
        img_id = self.ids[idx]
        
        img_info = self.coco.loadImgs(img_id)[0]
        img_path = os.path.join(self.root_dir, img_info['file_name'])
        
        img = Image.open(img_path).convert('RGB')
        orig_width, orig_height = img.size
        img = img.resize(self.img_size, Image.BILINEAR)
        
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)
        
        masks = []
        boxes = []
        labels = []
        
        for ann in anns:
            cat_id = ann['category_id']
            labels.append(cat_id)
            
            if type(ann['segmentation']) == list:  
                rles = coco_mask.frPyObjects(ann['segmentation'], orig_height, orig_width)
                mask = coco_mask.decode(rles)
                if len(mask.shape) > 2:
                    mask = mask.sum(axis=2) > 0
            else:  
                mask = coco_mask.decode(ann['segmentation'])
            
            mask = cv2.resize(mask.astype(np.uint8), self.img_size, interpolation=cv2.INTER_NEAREST)
            masks.append(mask)
            
            bbox = ann['bbox']  
            
            x_center = (bbox[0] + bbox[2] / 2) * self.img_size[0] / orig_width
            y_center = (bbox[1] + bbox[3] / 2) * self.img_size[1] / orig_height
            w = bbox[2] * self.img_size[0] / orig_width
            h = bbox[3] * self.img_size[1] / orig_height
            
            boxes.append([x_center, y_center, w, h])
        
        if self.transform:
            img = self.transform(img)
        else:
            img = transforms.ToTensor()(img)
        
        if masks:
            masks = torch.as_tensor(np.stack(masks), dtype=torch.uint8)
        else:
            masks = torch.zeros((0, self.img_size[1], self.img_size[0]), dtype=torch.uint8)

        if boxes:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
        else:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
        
        labels = torch.as_tensor(labels, dtype=torch.int64) if labels else torch.zeros(0, dtype=torch.int64)
        
        target = {
            'boxes': boxes,
            'labels': labels,
            'masks': masks,
            'image_id': torch.tensor([img_id], dtype=torch.int64),
            'orig_size': torch.as_tensor([orig_height, orig_width], dtype=torch.int64)
        }


        return img, target

def collate_fn_images(batch):
    images = []
    targets = []
    for img, tgt in batch:
        images.append(img)
        targets.append(tgt)
    
    images = torch.stack(images, dim=0)
    return images, targets

def get_coco_dataloaders(root_dir, batch_size=4, num_workers=0, shuffle=True, img_size=1024):
    """Create train/valid/test dataloaders for COCO format segmentation dataset.
    """

    try:
        with open(os.path.join(root_dir, TRAIN_ANNOTATION_FILE), "r") as f:
            train_data = json.load(f)

            # must have background class, so add 1
            num_classes = len(train_data['categories']) + 1
    except FileNotFoundError:
        raise FileNotFoundError(f"{TRAIN_ANNOTATION_FILE} not found in {root_dir}")

    # TODO: Add normalization 
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_dataset = COCOSegmentationDataset(
        root_dir=root_dir,
        ann_file=TRAIN_ANNOTATION_FILE,
        transform=transform,
        img_size=img_size
    )
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn_images
    )
    print(train_dataloader.__dict__)

    valid_dataset = COCOSegmentationDataset(
        root_dir=root_dir,
        ann_file=VALID_ANNOTATION_FILE,
        transform=transform,
        img_size=img_size
    )
    
    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn_images
    )
    
    test_dataset = COCOSegmentationDataset(
        root_dir=root_dir,
        ann_file=TEST_ANNOTATION_FILE,
        transform=transform,
        img_size=img_size
    )
    
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn_images
    )

    return train_dataloader, valid_dataloader, test_dataloader, num_classes
