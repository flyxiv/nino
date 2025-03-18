"""DataLoader for sprite/portrait images

Crops images starting from center.
"""

import pathlib
import os
import logging

from torch.utils.data import Dataset
from PIL import Image 
import torchvision.transforms as transforms
from .constants import INPUT_IMG_SIZE, OUTPUT_IMG_SIZE, VALID_IMG_EXTENSIONS
from .transform import INPUT_IMG_TRANSFORMS, OUTPUT_IMG_TRANSFORMS

class ImgDataset(Dataset):
    def __init__(self, input_dir, label_dir, device, /, *, input_img_size=INPUT_IMG_SIZE, output_img_size=OUTPUT_IMG_SIZE):
        self.input_dir = input_dir
        self.label_dir = label_dir

        self.input_img_size = input_img_size
        self.output_img_size = output_img_size

        if input_dir and label_dir:
            img_keys = {file for file in os.listdir(label_dir)}.intersection({file for file in os.listdir(input_dir)}) 

            self.label_images = [file for file in img_keys if os.path.splitext(file)[1].lower() in VALID_IMG_EXTENSIONS] 
            self.input_images = [file for file in img_keys if os.path.splitext(file)[1].lower() in VALID_IMG_EXTENSIONS]

        self.device = device

    def __len__(self):
        return len(self.label_images)

    def __getitem__(self, idx):
        img_name = self.input_images[idx]
        label_img_name = self.label_images[idx]

        input_img = Image.open(pathlib.Path(self.input_dir) / img_name) 
        label_img = Image.open(pathlib.Path(self.label_dir) / label_img_name)

        input_tensor = INPUT_IMG_TRANSFORMS(input_img).to(self.device)
        label_tensor = OUTPUT_IMG_TRANSFORMS(label_img).to(self.device)

        return input_tensor, label_tensor
