#!/bin/bash

file_path=$1
python scripts/collect_sprites.py --model ./trained_models/nino_seg_maskrcnn_e500.pth --config-file ./model/segmentation/mask_rcnn_resnet50/resnet_config.py --input-path $file_path --model-type maskrcnn --output-dir ./output_data/sprites