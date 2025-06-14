$image_name = $args[0]
$input_path = $args[1]
$output_dir = $args[2]

# Docker 실행 명령
docker run --gpus all -it `
  -v "${PWD}/trained_models:/app/nino/trained_models" `
  -v "${PWD}/$output_dir:/app/nino/results" `
  -v "${PWD}/$input_path:/app/nino/input" `
  "$image_name" /app/nino/scripts/collect_sprites.py --input-path /app/nino/input `
  --config-file /app/nino/model/segmentation/mask_rcnn_resnet50/resnet_config.py `
  --model /app/nino/trained_models/nino_seg_maskrcnn_e500.pth --model-type maskrcnn `
  --classification-model /app/nino/trained_models/nino_classification_efficientnet_v2_l.pth --output-dir /app/nino/results