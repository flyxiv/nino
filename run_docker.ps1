$image_name = $args[0]

# Docker 실행 명령
docker run --gpus all -it `
  -v ${PWD}/data/instance_segmentation_coco:/nino/data `
  -v ${PWD}/nino_seg_maskrcnn_e500.pth:/nino/nino_seg_maskrcnn_e500.pth `
  -v ${PWD}/output_data:/nino/output_data `
  "$image_name" bash