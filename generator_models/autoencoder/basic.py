"""Basic Pytorch Autoencoder Model.

Testing
"""

from torch import nn
from ..util.constants import INPUT_IMG_SIZE, OUTPUT_IMG_SIZE
import torch

class BasicAutoEncoder(nn.Module):
    """Basic Autoencoder Model to generate images.
 
    Only linear encoding/decoding. Probably will not be a good generator, but a good starting point.
    """
    def __init__(self, n_batch, input_img_size=INPUT_IMG_SIZE, output_img_size=OUTPUT_IMG_SIZE):
        super(BasicAutoEncoder, self).__init__()
        self.n_batch = n_batch
        self.input_img_size = input_img_size
        self.output_img_size = output_img_size 

        self.encoder = nn.Sequential(
            nn.Linear(self.input_img_size[0] * self.input_img_size[1] * 3, 1000),
            nn.LeakyReLU(0.1),
            nn.Linear(1000, 500),
            nn.LeakyReLU(0.1),
            nn.Linear(500, 100),
            nn.LeakyReLU(0.1),
        )

        self.decoder = nn.Sequential(
            nn.Linear(100, 500),
            nn.LeakyReLU(0.1),
            nn.Linear(500, 2000),
            nn.LeakyReLU(0.1),
            nn.Linear(2000, self.output_img_size[0] * self.output_img_size[1] * 3),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        x = self.encoder(x)
        x = self.decoder(x)
        x = x.reshape((self.n_batch, 3, self.output_img_size[0], self.output_img_size[1]))
        return x

    def predict(self, x):
        x = torch.flatten(x, start_dim=1)
        x = self.encoder(x)
        x = self.decoder(x)
        x = x.reshape((1, 3, self.output_img_size[0], self.output_img_size[1]))
        return x