import torch
import torch.nn as nn
import torchvision
import argparse
import optuna
from torch.utils.data import DataLoader
import torch.optim as optim
from torchvision import transforms
from model.sprite_classification.dataset import SpriteClassificationDataTable
from util.sprite_classifications import SPRITE_IDS
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from tqdm import tqdm
from plotly.io import show
from model.sprite_classification.consts import PREPROCESS_TRANSFORMS, SPRITE_IMG_SIZE

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, required=True, help='Path to the data directory')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    return parser.parse_args()

def objective(trial):
    transform = PREPROCESS_TRANSFORMS

    [train_dataset, val_dataset, test_dataset] = SpriteClassificationDataset.create_from_dir(args.data_dir, transform)

    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = torchvision.models.efficientnet_b6(weights=True)
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, len(SPRITE_IDS))
    model.to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer_name = trial.suggest_categorical('optimizer', ['AdamW', 'NAdam', 'Adam', 'RMSprop'])
    lr = trial.suggest_float('lr', 1e-5, 0.005, log=True)
    optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr)

    start_factor1 = trial.suggest_float('start_factor1', 0.1, 0.5)

    scheduler1 = LinearLR(optimizer, start_factor=start_factor1, end_factor=1.0, total_iters=5)
    t_max = trial.suggest_int('t_max', 5, 50)
    scheduler2 = CosineAnnealingLR(optimizer, T_max=t_max)

    scheduler = SequentialLR(optimizer, [scheduler1, scheduler2], milestones=[5])

    for epoch in range(args.epochs):
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

        if trial.should_prune():
            raise optuna.TrialPruned()

        precision = Precision(num_classes=len(SPRITE_IDS), average='micro', task='multiclass').to(device)
        recall = Recall(num_classes=len(SPRITE_IDS), average='micro', task='multiclass').to(device)

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                binary_predictions = torch.argmax(outputs, dim=1)

                precision.update(binary_predictions, labels)
                recall.update(binary_predictions, labels)

        trial.report(precision.compute() + recall.compute(), epoch)
        scheduler.step()

    return precision.compute() + recall.compute()
        
if __name__ == '__main__':
    args = parse_args()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    sampler = optuna.samplers.TPESampler(seed=10)
    study = optuna.create_study(direction='maximize', sampler=sampler)
    study.optimize(objective, n_trials=100)

    pruned_trials = study.get_trials(deepcopy=False, states=[optuna.trial.TrialState.PRUNED]) 
    complete_trials = study.get_trials(deepcopy=False, states=[optuna.trial.TrialState.COMPLETE])

    print("Study statistics: ")
    print("  Number of finished trials: ", len(complete_trials))
    print("  Number of pruned trials: ", len(pruned_trials))

    importance = optuna.importance.get_param_importances(study) 
    print(importance)

    print("Best trial:")
    trial = study.best_trial
    print(f"  Value: {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
