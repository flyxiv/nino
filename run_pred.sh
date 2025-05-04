#!/bin/bash

python scripts/collect_sprites.py --model ./nino_seg_maskrcnn_e500.pth --config-file ./model/segmentation/mask_rcnn_resnet50/resnet_config.py --input-path ./data/coco/test2017/266cfc09-rpg39.png --model-type maskrcnn --output-dir ./output_data/sprites