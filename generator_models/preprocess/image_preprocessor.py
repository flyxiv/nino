"""Preprocessing module to clip outside image to a predefined image size. 
"""

import cv2
import torch

class ImagePreprocessor:
	"""Preprocess image to match size of the model input.
	"""

	def __init__(self, image_size=256):
		self.image_size = image_size

	def _crop_image(self, image: tf.Tensor) -> tf.Tensor:


