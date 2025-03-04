import matplotlib.pyplot as plt
import torchvision

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

def plot_img_list(img_list):
	"""Plots multiple images in a grid.
	"""

	imshow(torchvision.utils.make_grid(img_list))
