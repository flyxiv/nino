import torch
import torch.nn as nn
import torchvision
import argparse
import optuna
from pathlib import Path
from torch.utils.data import DataLoader
import torch.optim as optim
from torchvision import transforms
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from tqdm import tqdm
from torchmetrics import Precision, Recall
from model.duplicate_detection.consts import PREPROCESS_TRANSFORMS
from model.duplicate_detection.dataset import DuplicateDetectionDataset, collate_duplicate_detection_fn
from model.duplicate_detection.efficientnet.duplicate_detection_model import DuplicateDetectionModel


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Config:
    data_dir = Path('./output_data/labels/duplicate_detection')
    batch_size = 16 
    lr =  0.0009253114358625602

    start_factor1 =  0.27416880209259337

    optimizer_name = 'NAdam' 
    t_max = 17 

    NO_UPDATE_EPOCHS_STOP = 50

[train_dataset, val_dataset, test_dataset] = DuplicateDetectionDataset.create_from_dir(Config.data_dir, PREPROCESS_TRANSFORMS)

train_loader = DataLoader(train_dataset, batch_size=Config.batch_size, shuffle=True, collate_fn=collate_duplicate_detection_fn)
val_loader = DataLoader(val_dataset, batch_size=Config.batch_size, shuffle=False, collate_fn=collate_duplicate_detection_fn)

model = DuplicateDetectionModel()
model.to(device)

criterion = nn.BCELoss()

optimizer = getattr(optim, Config.optimizer_name)(model.parameters(), lr=Config.lr)


scheduler1 = LinearLR(optimizer, start_factor=Config.start_factor1, end_factor=1.0, total_iters=5)
scheduler2 = CosineAnnealingLR(optimizer, T_max=Config.t_max)

scheduler = SequentialLR(optimizer, [scheduler1, scheduler2], milestones=[5])

best_precision = 0
no_update_epochs = 0
best_weights = None

for epoch in tqdm(range(200)):
    model.train()
    running_loss = 0.0

    for (image1, image2, label1_clarity, label2_clarity, label_similarity) in train_loader:
        image1 = image1.to(device)
        image2 = image2.to(device)
        label1_clarity = label1_clarity.to(device)
        label2_clarity = label2_clarity.to(device)
        label_similarity = label_similarity.to(device)

        optimizer.zero_grad()
        image1_clarity, image2_clarity, similarity = model(image1, image2)
        loss = criterion(image1_clarity, label1_clarity) + criterion(image2_clarity, label2_clarity) + criterion(similarity, label_similarity)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    
    model.eval()
    print(f'Loss: {running_loss / len(train_loader)}')

    precision_similarity = Precision(num_classes=2, average='micro', task='multiclass').to(device)
    recall_similarity = Recall(num_classes=2, average='micro', task='multiclass').to(device)

    with torch.no_grad():
        for image1, image2, label1_clarity, label2_clarity, label_similarity in val_loader:
            image1 = image1.to(device)
            image2 = image2.to(device)
            label1_clarity = label1_clarity.to(device)
            label2_clarity = label2_clarity.to(device)
            label_similarity = label_similarity.to(device)

            image1_clarity, image2_clarity, similarity = model(image1, image2)

            similarity_pred = torch.where(similarity > 0.5, 1, 0)

            precision_similarity.update(similarity_pred, label_similarity)
            recall_similarity.update(similarity_pred, label_similarity)

    final_precision_similarity = precision_similarity.compute() 
    final_recall_similarity = recall_similarity.compute()

    print(f'Epoch {epoch} [Similarity] Precision: {final_precision_similarity} - Recall: {final_recall_similarity}')

    scheduler.step()

    if final_precision_similarity > best_precision:
        best_precision = final_precision_similarity
        no_update_epochs = 0
        best_weights = model.state_dict()
    else:
        no_update_epochs += 1

    if no_update_epochs > Config.NO_UPDATE_EPOCHS_STOP:
        break

    precision_similarity.reset()
    recall_similarity.reset()


torch.save(best_weights, 'model.pth')