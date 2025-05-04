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
    
    # Visualize results
    plt.figure(figsize=(12, 8))
    if masks is not None:
        # Draw masks
        for i, mask in enumerate(masks):
            if scores[i] > 0.5:  # Only show results with confidence > 0.5
                # Create a colored overlay for the mask
                color = np.random.rand(3)
                mask_img = np.zeros_like(img)
                for c in range(3):
                    mask_img[:, :, c] = mask * 255 * color[c]
                # Apply the mask over the image with transparency
                img = cv2.addWeighted(img, 1, mask_img.astype(np.uint8), 0.5, 0)
    
    # Draw bounding boxes
    for i, bbox in enumerate(bboxes):
        if scores[i] > 0.5:  # Only show results with confidence > 0.5
            x1, y1, x2, y2 = bbox.astype(int)
            label_text = f"{model.dataset_meta['classes'][labels[i]]}: {scores[i]:.2f}"
            color = (255, 0, 0)  # Red for bounding boxes
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img, label_text, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    plt.imshow(img)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('result.png')
    plt.show()
    
    # Save result to file
    cv2.imwrite('result_output.jpg', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    
    print("Inference completed successfully!")

if __name__ == '__main__':
    args = parse_args()
    main(args.config, args.checkpoint, args.image)