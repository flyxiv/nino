from torchvision import transforms

SPRITE_IMG_SIZE = (50, 100)

PREPROCESS_TRANSFORMS = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(SPRITE_IMG_SIZE),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

