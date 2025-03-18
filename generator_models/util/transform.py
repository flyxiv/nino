from torchvision.transforms import transforms
from .constants import INPUT_IMG_SIZE, OUTPUT_IMG_SIZE

INPUT_IMG_TRANSFORMS = transforms.Compose([
    transforms.CenterCrop(INPUT_IMG_SIZE),
    transforms.ToTensor()
])

OUTPUT_IMG_TRANSFORMS = transforms.Compose([
    transforms.CenterCrop(OUTPUT_IMG_SIZE),
    transforms.ToTensor()
])
