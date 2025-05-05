from torchvision import transforms

SPRITE_IMG_SIZE = (80, 40)


PREPROCESS_TRANSFORMS = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(SPRITE_IMG_SIZE),
])