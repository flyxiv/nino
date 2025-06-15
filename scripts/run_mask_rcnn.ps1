# Check if required arguments are provided
if ($args.Count -lt 3) {
    Write-Host "Usage: .\run_mask_rcnn.ps1 <image_name> <input_path> <output_dir>"
    Write-Host "Example: .\run_mask_rcnn.ps1 nino ./input ./results"
    exit 1
}

$image_name = $args[0]
$input_path = $args[1]
$output_dir = $args[2]

# Check if input path exists
if (-not (Test-Path $input_path)) {
    Write-Host "Error: Input path '$input_path' does not exist"
    exit 1
}

# Create output directory if it doesn't exist
if (-not (Test-Path $output_dir)) {
    New-Item -ItemType Directory -Path $output_dir | Out-Null
}

# Docker 실행 명령
docker run --gpus all `
  -v "${PWD}/trained_models:/app/nino/trained_models" `
  -v "${PWD}/${output_dir}:/app/nino/results" `
  -v "${PWD}/${input_path}:/app/nino/input" `
  -e PYTHONPATH="/app/nino" `
  "$image_name" python /app/nino/scripts/collect_sprites.py `
    --input-path /app/nino/input `
    --config-file /app/nino/model/segmentation/mask_rcnn_resnet50/resnet_config.py `
    --model /app/nino/trained_models/nino_seg_maskrcnn_e500.pth `
    --model-type maskrcnn `
    --classification-model /app/nino/trained_models/nino_classification_efficientnet_v2_l.pth `
    --output-dir /app/nino/results