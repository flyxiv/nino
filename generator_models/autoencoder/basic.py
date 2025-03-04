"""Basic Pytorch Autoencoder Model.
"""

from torch import nn
import torch

class BasicAutoencoder(nn.Module):
    """Basic Autoencoder Model to generate images.
 
    Only handles square shape images (could preprocess to square).
    Only linear encoding/decoding. Probably will not be a good generator, but a good starting point.
    """
    def __init__(self, n_batch, image_size=28):
        super(BasicGAN, self).__init__()
        self.n_batch = n_batch
        self.image_size = image_size
        self.loss_fn = nn.L1Loss()

        self.encoder = nn.Sequential(
            nn.Linear(self.image_size ** 2, 200),
            nn.LeakyReLU(0.1),
            nn.Linear(200, 100),
            nn.LeakyReLU(0.1),
            nn.Linear(100, 30),
            nn.LeakyReLU(0.1),
        )

        self.decoder = nn.Sequential(
            nn.Linear(30, 100),
            nn.LeakyReLU(0.1),
            nn.Linear(100, 200),
            nn.LeakyReLU(0.1),
            nn.Linear(200, self.image_size ** 2),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        x = self.encoder(x)
        x = self.decoder(x)
        x = x.reshape((self.n_batch, 1, self.image_size, self.image_size))
        return x

    def predict(self, x):
        x = torch.flatten(x, start_dim=1)
        x = self.encoder(x)
        x = self.decoder(x)
        x = x.reshape((1, self.image_size, self.image_size))
        return x

    def train(self, data_loader, optimizer, num_epochs=25):
        train_model(self, data_loader, optimizer=optimizer, loss_fn=self.loss_fn, num_epochs=num_epochs)
