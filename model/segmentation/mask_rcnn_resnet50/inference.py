from util.segmented_model_output import SegmentedModelOutput
from mmdet.apis import init_detector, inference_detector
import numpy as np

def create_from_maskrcnn_output(mask, bbox, label, score):
    bbox = bbox.cpu().numpy()
    label = label.cpu().numpy()
    score = score.cpu().numpy()
    mask = mask.cpu().numpy().astype(np.float32) 

    return SegmentedModelOutput(
        bbox_mask=mask,
        bbox_x1=bbox[0],
        bbox_y1=bbox[1],
        bbox_x2=bbox[2],
        bbox_y2=bbox[3],
        bbox_conf=score,
        bbox_class=label
    )

def create_segmented_model_outputs_maskrcnn(img_batch, model, conf_threshold: float = 0.9):
    """Collect model outputs and their corresponding images from a single image or a batch of images.
    """
    batch_results = inference_detector(model, img_batch)  
    model_outputs = []

    if type(batch_results) == list:
        for result_idx, result in enumerate(batch_results):
            img = img_batch[result_idx]

            for bbox, label, score, mask in zip(result.pred_instances.bboxes, result.pred_instances.labels, result.pred_instances.scores, result.pred_instances.masks):
                if score > conf_threshold:
                    model_output = create_from_maskrcnn_output(mask, bbox, label, score)
                    model_outputs.append((model_output, img))

    else:
        img = img_batch
        for bbox, label, score, mask in zip(batch_results.pred_instances.bboxes, batch_results.pred_instances.labels, batch_results.pred_instances.scores, batch_results.pred_instances.masks):
            if score > conf_threshold:
                model_output = create_from_maskrcnn_output(mask, bbox, label, score)
                model_outputs.append((model_output, img_batch))

    return model_outputs