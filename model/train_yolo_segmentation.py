"""Train YOLO Segmentation Model Using Custom Dataset

"""

import logging
import argparse

from pathlib import Path
from ultralytics import YOLO

def train_yolo_segmentation_model(*, num_epochs, data_dir):
    """train YOLOv11 segmentation model
    """
    model = YOLO('yolo11s-seg.pt') 

    dataset_metadata_file_path = Path(data_dir) / 'data.yaml' 

    model.train(data=dataset_metadata_file_path, epochs=num_epochs, imgsz=640, device=0)
        

if __name__ == '__main__':
    logging.basicConfig(
        format='%(asctime)s %(levelname)s:%(message)s',
        level=logging.INFO,
        datefmt='%m/%d/%Y %I:%M:%S %p',
    )

    parser = argparse.ArgumentParser()

    parser.add_argument('--num-epochs', type=int, required=False, default=100, help='Number of epochs to train')
    parser.add_argument('--data-dir', type=str, required=True, help='input dataset directory')

    args = parser.parse_args()

    num_epochs = args.num_epochs
    data_dir = args.data_dir

    train_yolo_segmentation_model(num_epochs=num_epochs, data_dir=data_dir)