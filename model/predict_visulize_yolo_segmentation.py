"""Predict Sample Images with Trained YOLO Segmentation Model and Visualize

usage:

```sh
$ python -m segmentation_models.predict_visulize_yolo_segmentation --trained-model-path ..\yolov5\runs\segment\train8\weights\best.pt --sample-img-path .\test.png
```
"""

import logging
import argparse

from ultralytics import YOLO

if __name__ == '__main__':
    logging.basicConfig(
        format='%(asctime)s %(levelname)s:%(message)s',
        level=logging.INFO,
        datefmt='%m/%d/%Y %I:%M:%S %p',
    )

    parser = argparse.ArgumentParser()

    parser.add_argument('--trained-model-path', type=str, required=True, help='pretrained .pt file')
    parser.add_argument('--sample-img-path', type=str, required=True, help='sample image data path')

    args = parser.parse_args()

    trained_model_path = args.trained_model_path
    sample_img_path = args.sample_img_path

    model = YOLO(trained_model_path) 

    results = model(sample_img_path) 

    for result in results:
        boxes = result.boxes  # Boxes object for bounding box outputs
        masks = result.masks  # Masks object for segmentation masks outputs
        keypoints = result.keypoints  # Keypoints object for pose outputs
        probs = result.probs  # Probs object for classification outputs
        obb = result.obb  # Oriented boxes object for OBB outputs
        result.show()  # display to screen
        result.save(filename="result.jpg")