#!/bin/bash

python scripts/collect_sprites.py --model ./nino_seg_maskrcnn_e500.pth --checkpoint_file ./model/segmentation/resnet50_mask_rcnn/resnet_config.py --input_path ./data/coco/test2017/266cfc09-rpg39.png --model-type maskrcnn 