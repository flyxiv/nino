#!/bin/bash

python scripts/collect_sprites.py --model trained_models/nino_seg_maskrcnn_e500.pth --config-file model/segmentation/mask_rcnn_resnet50/resnet_config.py --model-type maskrcnn --output-dir ./output_data --input-path ./input_files