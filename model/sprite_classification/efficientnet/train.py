import torch
import torch.nn as nn
import torchvision
import argparse
import optuna
from torch.utils.data import DataLoader
import torch.optim as optim
from torchvision import transforms
from model.sprite_classification.dataset import SpriteClassificationDataset
from util.sprite_metadata import SPRITE_NO_DIRECTION_IDS, SPRITE_NO_DIRECTIONS
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from tqdm import tqdm
from model.sprite_classification.consts import PREPROCESS_TRANSFORMS, SPRITE_IMG_SIZE
from torchmetrics import Precision, Recall

data_dir = './output_data/labels/classification'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

[train_dataset, val_dataset, test_dataset] = SpriteClassificationDataset.create_from_dir(data_dir, PREPROCESS_TRANSFORMS)

batch_size = 16 
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

weights = torchvision.models.EfficientNet_V2_L_Weights.DEFAULT
model = torchvision.models.efficientnet_v2_l(weights=weights)
model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, len(SPRITE_NO_DIRECTION_IDS))
model.to(device)

criterion = nn.CrossEntropyLoss()

lr =  0.0009253114358625602
optimizer_name = 'NAdam' 
optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr)

start_factor1 =  0.27416880209259337

scheduler1 = LinearLR(optimizer, start_factor=start_factor1, end_factor=1.0, total_iters=5)
t_max = 17 
scheduler2 = CosineAnnealingLR(optimizer, T_max=t_max)

scheduler = SequentialLR(optimizer, [scheduler1, scheduler2], milestones=[5])

best_precision = 0
no_update_epochs = 0
NO_UPDATE_EPOCHS_STOP = 50

for epoch in tqdm(range(200)):
    model.train()
    running_loss = 0.0

    for (images, labels) in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    
    model.eval()
    print(f'Loss: {running_loss / len(train_loader)}')

    precision = Precision(num_classes=len(SPRITE_NO_DIRECTION_IDS), average='micro', task='multiclass').to(device)
    precision_per_class = Precision(num_classes=len(SPRITE_NO_DIRECTION_IDS), average=None, task='multiclass').to(device)
    recall = Recall(num_classes=len(SPRITE_NO_DIRECTION_IDS), average='micro', task='multiclass').to(device)

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            binary_predictions = torch.argmax(outputs, dim=1)

            precision_per_class.update(binary_predictions, labels)
            precision.update(binary_predictions, labels)
            recall.update(binary_predictions, labels)

    final_precision = precision.compute() 
    final_recall = recall.compute()

    print(f'Epoch {epoch} Precision: {final_precision} - Recall: {final_recall}')

    scheduler.step()

    if final_precision > best_precision:
        best_precision = final_precision
        no_update_epochs = 0
    else:
        no_update_epochs += 1

    if no_update_epochs > NO_UPDATE_EPOCHS_STOP:
        break

    for i, p in enumerate(precision_per_class.compute()):
        print(f'{SPRITE_NO_DIRECTIONS[i]}: {p}')

    precision.reset()
    recall.reset()
    precision_per_class.reset()

torch.save(model.state_dict(), 'model.pth')