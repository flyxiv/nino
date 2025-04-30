"""Augment image size to 

Puts image to the center, and pads image with blank background to make 512x512 image

run example: in nino base directory,
python -m preprocess.change_image_size --image-dir ./data/input --output-dir ./data/output
"""

import argparse
import os
from pathlib import Path
from PIL import Image

DIFFUSION_IMG_SIZE = (512, 512)
TRANSPARENT_COLOR = (255, 255, 255, 255)


def _preprocess_image_to_correct_size(img):
    # if image's width and height are both smaller than wanted output size -> create blank image with correct size and paste image to center
    if img.width <= DIFFUSION_IMG_SIZE[0] and img.height <= DIFFUSION_IMG_SIZE[1]:
        new_img_with_correct_size = Image.new(
            'RGBA', DIFFUSION_IMG_SIZE, TRANSPARENT_COLOR)

        new_img_center_x = int(DIFFUSION_IMG_SIZE[0] / 2 - img.width / 2)
        new_img_center_y = int(DIFFUSION_IMG_SIZE[1] / 2 - img.height / 2)

        new_img_with_correct_size.paste(
            im=img, box=(new_img_center_x, new_img_center_y)
        )

        return new_img_with_correct_size
    else:
        img.resize(DIFFUSION_IMG_SIZE)
        return img


def pad_images_to_diffusion_input_size(img_dir, output_dir):
    img_dir_path = Path(img_dir)
    imgs = img_dir_path.glob('**/*')

    os.makedirs(output_dir, exist_ok=True)

    for img_path in imgs:
        img_name = img_path.name
        img = Image.open(img_path)

        if img.mode != 'RGBA':
            img = img.convert('RGBA')

        new_img_with_correct_size = _preprocess_image_to_correct_size(img)

        output_path = os.path.join(output_dir, img_name)
        new_img_with_correct_size.save(output_path, 'PNG')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img-dir', type=str, default=True,
                        help='directory with images to be preprocessed.')
    parser.add_argument('--output-dir', type=str, default=True,
                        help='directory where preprocessed image files will be saved')

    args = parser.parse_args()

    pad_images_to_diffusion_input_size(
        img_dir=args.img_dir, output_dir=args.output_dir)
