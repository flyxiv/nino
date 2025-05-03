# run ex)
# ./scripts/upload_to_google_cloud.ps1 E:\nino\data\instance_segmentation_yolo\train junyeopn_dataset instance_segmentation_yolo

$localDirectory = $args[0]
$bucketName = $args[1]
$prefix = $args[2]

gcloud storage cp --recursive $localDirectory gs://$bucketName/$prefix