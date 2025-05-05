import cv2
from PIL import Image
import numpy as np
import logging
import os

from model.segmentation.yolo_seg.inference import create_segmented_model_outputs_yolo

    
def collect_sprites_from_images(img_batch, output_dir: str, model, model_type: str, frame_idx = None, conf_threshold: float = 0.9):
    """Collect sprites from a single image or a batch of images.
    """

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    if model_type == 'yolo':
        model_outputs = create_segmented_model_outputs_yolo(img_batch, model, conf_threshold)
    else:
        from model.segmentation.mask_rcnn_resnet50.inference import create_segmented_model_outputs_maskrcnn
        model_outputs = create_segmented_model_outputs_maskrcnn(img_batch, model, conf_threshold)

    for sprite_cnt, (model_output, img) in enumerate(model_outputs):
        segmented_sprite = get_segmented_sprite(img, model_output)

        if frame_idx is not None:
            file_name = f"sprite_{frame_idx}_{sprite_cnt}_{int(model_output.bbox_conf*100)}.png"
        else:
            file_name = f"sprite_{sprite_cnt}_{int(model_output.bbox_conf*100)}.png"

        output_filename = os.path.join(
            output_dir, 
            file_name
        )

        sprite_image = Image.fromarray(segmented_sprite)
        sprite_image.save(output_filename)
        logging.info(f"Saved masked crop {sprite_cnt} to {output_filename}")
    
    logging.info(f"All masked crops saved to {output_dir}")   


def get_segmented_sprite(original_image, model_output, model='yolo_seg'):
    """Crops bounding box of sprites from original image and saves only valid mask pixels. 0 masks are translated to white background. 

    Args:
        original_image: PIL.Image.Image
        model_output: SegmentedModelOutput
        model: str

    Returns:
        PIL.Image.Image
    """
    img_width = original_image.shape[1]
    img_height = original_image.shape[0]

    # 박스 영역을 안전하게 자르기 (경계 벗어남 방지)
    x1 = int(max(0, model_output.bbox_x1))
    y1 = int(max(0, model_output.bbox_y1))
    x2 = int(min(img_width, model_output.bbox_x2))
    y2 = int(min(img_height, model_output.bbox_y2))
    
    
    # 마스크를 박스 크기에 맞게 크롭 및 리사이즈
    mask_resized = cv2.resize(
        model_output.bbox_mask.astype(np.float32),
        (img_width, img_height),
        interpolation=cv2.INTER_LINEAR
    )
    
    # 이진 마스크로 변환 (1: 객체, 0: 배경)
    mask_binary = (mask_resized > 0.5).astype(np.uint8)
    
    # 마스크 적용 (마스크=1: 원본 이미지, 마스크=0: 흰색 배경)
    white_background = np.ones_like(original_image) * 255
    masked_crop = np.zeros_like(original_image)
    
    for c in range(3):  # RGB 채널
        masked_crop[:, :, c] = np.where(mask_binary == 1, 
                                        original_image[:, :, c], 
                                        white_background[:, :, c])

    return masked_crop[y1:y2, x1:x2].copy()
    
