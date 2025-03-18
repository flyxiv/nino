"""Util function for training AutoEncoder PyTorch Models.
"""

import torch
import logging
import argparse

from tqdm import tqdm
from .basic import BasicAutoEncoder
from .cnn_encoder import CNNAutoEncoder
from ..util.img_dataloader import ImgDataset
from torch.optim import Adam
from torch.nn import L1Loss
from torch.utils.data import DataLoader
from torch.jit import script

from torch.nn.functional import l1_loss 
from ..util.constants import EPSILON

def masked_loss(pred, target, mask):
    element_wise_loss = l1_loss(pred, target)
    masked_loss = element_wise_loss * mask

    return masked_loss.sum() / (mask.sum() + EPSILON)
    

def train_autoencoder_model(model, data_loader, /, *, optimizer, loss_fn, num_epochs=200):
	for epoch in range(num_epochs):
		running_loss = 0.0

		for i, data in tqdm(enumerate(data_loader)):
			inputs, labels = data
			mask = (labels < 0.95).float()

			optimizer.zero_grad()

			outputs = model(inputs)

			loss = masked_loss(outputs, labels, mask)
			loss.backward()

			optimizer.step()

			running_loss += loss.item()

		print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
		running_loss = 0.0

	print('Finished Training')


if __name__ == '__main__':
    logging.basicConfig(
        format='%(asctime)s %(levelname)s:%(message)s',
        level=logging.INFO,
        datefmt='%m/%d/%Y %I:%M:%S %p',
    )

    parser = argparse.ArgumentParser()

    parser.add_argument('--model', type=str, required=False, default='basic', help='choose the type of autoencoder that will be trained')
    parser.add_argument('--input-dir', required=False, help='input image for generation')
    parser.add_argument('--label-dir', required=False, help='input image for generation')


    args = parser.parse_args()
    
    if args.model == 'basic':
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        dataset = ImgDataset(args.input_dir, args.label_dir, device)
        model = BasicAutoEncoder(1)
        model.to(device)
        data_loader = DataLoader(dataset, batch_size=1, shuffle=True)
        optimizer = Adam(model.parameters(), lr=0.001) 
        loss_fn = L1Loss() 

        train_autoencoder_model(model, data_loader, optimizer=optimizer, loss_fn=loss_fn)
        model_scripted = script(model)
        model_scripted.save('autoencoder.pt')

    if args.model == 'cnn':
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        dataset = ImgDataset(args.input_dir, args.label_dir, device, output_img_size=(79, 79))
        model = CNNAutoEncoder(1)
        model.to(device)
        data_loader = DataLoader(dataset, batch_size=1, shuffle=True)
        optimizer = Adam(model.parameters(), lr=0.0004) 
        loss_fn = L1Loss() 

        train_autoencoder_model(model, data_loader, optimizer=optimizer, loss_fn=loss_fn)
        model_scripted = script(model)
        model_scripted.save('cnn.pt')

