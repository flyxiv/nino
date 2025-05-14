import os

from util.segmented_model_output import SegmentedModelOutput


def create_from_yolo_output(mask, bbox):
    return SegmentedModelOutput(
        bbox_mask=mask,
        bbox_x1=bbox[0],
        bbox_y1=bbox[1],
        bbox_x2=bbox[2],
        bbox_y2=bbox[3],
        bbox_conf=bbox[4],
        bbox_class=bbox[5]
    )

def create_segmented_model_outputs_yolo(img_batch, model, conf: float = 0.9):
    """Collect model outputs and their corresponding images from a single image or a batch of images.
    """
    results = model(img_batch, device=0)  
    model_outputs = []

    for (result_idx, result) in enumerate(results):
        if result.masks is not None and result.boxes is not None:
            masks = result.masks.data.cpu().numpy()  
            boxes = result.boxes.data.cpu().numpy()  
            
            for mask, box in zip(masks, boxes):
                model_output = create_from_yolo_output(mask, box)
                if model_output.bbox_conf > conf:
                    if type(img_batch) == list:
                        img = img_batch[result_idx]
                    else:
                        img = img_batch
                    model_outputs.append((model_output, img))

    return model_outputs

 