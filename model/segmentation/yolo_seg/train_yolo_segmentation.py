"""Train YOLO Segmentation Model Using Custom Dataset

"""

import logging
import argparse

from pathlib import Path
from ultralytics import YOLO

def train_yolo_segmentation_model(*, num_epochs, data_dir, imgsz, batch_size, device):
    """train YOLOv8 segmentation model
    """
    model = YOLO('yolo11x-seg.pt') 

    dataset_metadata_file_path = Path(data_dir) / 'data.yaml' 

    model.train(data=dataset_metadata_file_path, epochs=num_epochs, imgsz=imgsz, batch=batch_size, device=device)

if __name__ == '__main__':
    logging.basicConfig(
        format='%(asctime)s %(levelname)s:%(message)s',
        level=logging.INFO,
        datefmt='%m/%d/%Y %I:%M:%S %p',
    )

    parser = argparse.ArgumentParser()

    parser.add_argument('--num-epochs', type=int, required=False, default=100, help='Number of epochs to train')
    parser.add_argument('--data-dir', type=str, required=True, help='input dataset directory')
    parser.add_argument('--imgsz', type=int, required=False, default=1024, help='image size')
    parser.add_argument('--batch-size', type=int, required=False, default=8, help='batch size')
    parser.add_argument('--device', type=str, required=False, default='0', help='device')

    args = parser.parse_args()

    num_epochs = args.num_epochs
    data_dir = args.data_dir
    imgsz = args.imgsz
    batch_size = args.batch_size
    device = args.device

    train_yolo_segmentation_model(num_epochs=num_epochs, data_dir=data_dir, imgsz=imgsz, batch_size=batch_size, device=device)