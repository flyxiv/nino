"""Basic Pytorch Autoencoder Model.

Testing
"""

from torch import nn
from ..util.constants import INPUT_IMG_SIZE, OUTPUT_IMG_SIZE
import torch

class CNNAutoEncoder(nn.Module):
    """CNN Autoencoder Model to generate images.
    """
    def __init__(self, n_batch, input_img_size=INPUT_IMG_SIZE, output_img_size=OUTPUT_IMG_SIZE):
        super(CNNAutoEncoder, self).__init__()
        self.n_batch = n_batch
        self.input_img_size = input_img_size
        self.output_img_size = output_img_size 

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=2),
            nn.LeakyReLU(0.1),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2),
            nn.LeakyReLU(0.1),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2),
            nn.LeakyReLU(0.1),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding='valid'),
            nn.LeakyReLU(0.1),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=3, stride=2),
            nn.LeakyReLU(0.1),
            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=3),
            nn.LeakyReLU(0.1),
            nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=3),
            nn.LeakyReLU(0.1),
            nn.ConvTranspose2d(in_channels=16, out_channels=3, kernel_size=2),
        )
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        x = x.reshape((self.n_batch, 3, self.output_img_size[0], self.output_img_size[1]))
        return x

    def predict(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        x = x.reshape((1, 3, self.output_img_size[0], self.output_img_size[1]))
        return x