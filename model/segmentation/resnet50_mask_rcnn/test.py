import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from mmengine.config import Config
from mmdet.apis import init_detector, inference_detector
from mmdet.visualization import visualize_det_bboxes

def parse_args():
    parser = argparse.ArgumentParser(description='Test the model')
    parser.add_argument('--config', type=str, default='./model/segmentation/resnet50_mask_rcnn/resnet_config.py', help='Path to the config file')
    parser.add_argument('--checkpoint', type=str, default='nino_segment_maskrcnn_e500.pth', help='Path to the checkpoint file')
    parser.add_argument('--image', type=str, default='test.jpg', help='Path to the image file')
    return parser.parse_args()

def main(config_file, checkpoint_file, image_path):
    # Specify the device to use (GPU or CPU)
    device = 'cuda:0'  # Use 'cpu' if you don't have GPU
    
    # Initialize the detector
    model = init_detector(config_file, checkpoint_file, device=device)
    
    # Run inference on a single image
    result = inference_detector(model, image_path)
    
    # Visualize results
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Get detection results
    bboxes = result.pred_instances.bboxes.cpu().numpy()
    labels = result.pred_instances.labels.cpu().numpy()
    scores = result.pred_instances.scores.cpu().numpy()
    
    # Get segmentation masks if available
    if hasattr(result.pred_instances, 'masks') and len(result.pred_instances.masks) > 0:
        masks = result.pred_instances.masks.cpu().numpy()
    else:
        masks = None

    print(bboxes)
    print(labels)
    print(scores)
    print(masks)
    

if __name__ == '__main__':
    args = parse_args()
    main(args.config, args.checkpoint, args.image)