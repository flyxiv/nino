$image_name = $args[0]

# Docker 실행 명령
docker run --gpus all -it `
  -v ${PWD}/data/instance_segmentation_coco:/nino/data `
  -v ${PWD}/trained_models:/nino/trained_models `
  -v ${PWD}/output_data:/nino/output_data `
  "$image_name" bash