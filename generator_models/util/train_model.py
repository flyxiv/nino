"""Util function for training PyTorch Models.
"""

import torch

def train_model(model, train_loader, /, *, optimizer, loss_fn, num_epochs=25):
	for epoch in range(num_epochs):
		running_loss = 0.0

		for i, data in enumerate(trainloader, 0):
			inputs, labels = data

			optimizer.zero_grad()

			outputs = model(inputs)
			loss = loss_fn(outputs, labels)
			loss.backward()
			optimizer.step()

			running_loss += loss.item()
			if i % 2000 == 1999:   
				print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
				running_loss = 0.0
