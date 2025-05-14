# Instance Segmentation + GAN Model for Generating 2D Sprites Based on Character Portraits

## Structure

![img](./img/diagram.png)

1. We make an instance segmentation model that will collect character portraits and 2D sprites to speed up data collection process.
2. img2img model to generate 2D sprites from character portraits. Start with `idle-downward` generation, and then expand to creating whole set of sprites.

## Environment
**Mask-RCNN needs old versions of python/pytorch/CUDA because of mmdetection(python 3.7, CUDA 11.6, ...)**
   * best to create Docker container in `./dockerfiles`

YOLO based and segmentation/duplicate detection model(efficientnet) need python 3.9 above.

## Start

```sh
python -m venv venv

# Build docker container
$ docker build -f ./dockerfiles/Dockerfile -t nino:v0.1 .

# windows powershell
$ . venv/Scripts/activate

# macOS/Linux
$ . venv/bin/activate


# setup environment
$ pip install -e .

# start label studio
$ label-studio start

# split YOLO
python -m scripts.split_yolo_exported_files --input-dir .\data\instance_segmentation_yolo --output-dir .\data\instance_segmentation_yolo

# train segmentation YOLO model
python -m segmentation_models.predict_visulize_yolo_segmentation --trained-model-path ..\yolov5\runs\segment\train8\weights\best.pt --sample-img-path .\data\instance_segmentation_yolo\valid\images\8d89728f-TWCI_2025_3_7_16_28_40.jpg
```


## Training

### 1. CO_DETR

* Setup docker 

```sh
$ docker build -t nino-detr:v0.1 .\dockerfiles\co-detr\

# check container ID with 'docker images'
$ docker run --gpus all -it -v .\data\instance_segmentation_coco:/Co-DETR/data -v .\model\segmentation\Co-DETR:/Co-DETR e8c478a237fd bash

# inside docker
$ python tools/get_checkpoint.py
$ mv checkpoint/pytorch_model.pth checkpoint/co_detr_pretrained.pth
$ python tools/train.py projects/configs/co_dino_vit/co_dino_5scale_vit_large_coco_instance.py

# inference 
$ python tools/test.py projects/configs/co_dino_vit/co_dino_5scale_vit_large_coco_instance.py ./checkpoints/co_detr_pretrained.pth --eval bbox --gpu-id 0 --cfg-options data.workers_per_gpu=0
```


```sh
docker run --gpus all -it -v .\data\instance_segmentation_coco:/nino/data -v .\nino_seg_maskrcnn_e500.pth:/nino/nino_seg_maskrcnn_e500.pth 2acb9278575c bash    
```