"""Predict Sample Images with Trained YOLO Segmentation Model and Visualize

usage:

```sh
$ python -m segmentation_models.predict_visulize_yolo_segmentation --trained-model-path ..\yolov5\runs\segment\train8\weights\best.pt --sample-img-path .\test.png
```
"""

import logging
import argparse
import numpy as np
import cv2
from PIL import Image
import os
from util.get_segmented_sprite import SegmentedModelOutput, get_segmented_sprite

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
    parser.add_argument('--output-dir', type=str, required=False, default='./', help='output directory')
    parser.add_argument('--separate-sprites', action='store_true', help='separate sprites')

    args = parser.parse_args()
    trained_model_path = args.trained_model_path
    sample_img_path = args.sample_img_path
    output_dir = args.output_dir
    
    # 출력 디렉토리가 없으면 생성
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 원본 이미지 로드
    original_image = cv2.imread(sample_img_path)
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    
    # 로그에 원본 이미지 크기 출력
    img_height, img_width = original_image.shape[:2]
    logging.info(f"Original image dimensions: {img_width}x{img_height}")
    
    # 모델 로드 및 예측 (원본 크기 그대로 예측하도록 설정)
    model = YOLO(trained_model_path)
    results = model(sample_img_path, device=0)  # 원본 이미지 크기로 설정

    for result_idx, result in enumerate(results):
        # 마스크와 박스가 있는지 확인
        if result.masks is not None and result.boxes is not None:
            masks = result.masks.data.cpu().numpy()  # 마스크 데이터
            boxes = result.boxes.data.cpu().numpy()  # 박스 데이터
            
            for mask_idx, (mask, box) in enumerate(zip(masks, boxes)):
                model_output = SegmentedModelOutput.create_from_yolo_output(mask, box)
                segmented_sprite = get_segmented_sprite(original_image, model_output)
                
                # 결과 저장
                output_filename = os.path.join(
                    output_dir, 
                    f"box_mask_{result_idx}_{mask_idx}_{int(model_output.bbox_conf*100)}.png"
                )
                
                # PNG 형식으로 저장
                sprite_image = Image.fromarray(segmented_sprite)
                sprite_image.save(output_filename)
                sprite_image.show()
                logging.info(f"Saved masked crop {mask_idx} to {output_filename}")
    
    logging.info(f"All masked crops saved to {output_dir}")