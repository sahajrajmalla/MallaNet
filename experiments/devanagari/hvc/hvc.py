# To run in Google Colab or single-GPU environment:
# 1. Install dependencies:
#    !pip install torch torchvision pandas scikit-learn
# 2. Run:
#    python experiment_single_gpu.py > output.log 2>&1
# Monitor logs: tail -f output.log

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
import os
from torch.amp import autocast, GradScaler
import logging
import sys
import time
import warnings
import json
import random
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix

warnings.filterwarnings("ignore", category=UserWarning)

# Setup logging
def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('training.log', mode='w'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger()

# Custom transform for data augmentation
class AddGaussianNoise(object):
    def __init__(self, mean=0., std=0.05):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

# Residual block with configurable dropout
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, dropout_rate=0.05):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout(dropout_rate)
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.dropout(out)
        out += identity
        out = self.relu(out)
        return out

# Homogeneous Vector Capsule layer
class HVCLayer(nn.Linear):
    def __init__(self, in_features, num_classes):
        super(HVCLayer, self).__init__(in_features, num_classes, bias=False)

# Merging layer for combining branch outputs
class MergingLayer(nn.Module):
    def __init__(self, num_branches=3):
        super(MergingLayer, self).__init__()
        self.w = nn.Parameter(torch.ones(num_branches) / num_branches)

    def forward(self, inputs):
        weights = F.softmax(self.w, dim=0)
        return sum(w * logit for w, logit in zip(weights, inputs))

# Enhanced BMCNN base with increased capacity
class BMCNNBase(nn.Module):
    def __init__(self, dropout_rate=0.05):
        super(BMCNNBase, self).__init__()
        self.conv_block1 = nn.Sequential(
            ResidualBlock(1, 128, stride=1, dropout_rate=dropout_rate),
            ResidualBlock(128, 128, stride=1, dropout_rate=dropout_rate),
            ResidualBlock(128, 128, stride=1, dropout_rate=dropout_rate)
        )
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv_block2 = nn.Sequential(
            ResidualBlock(128, 256, stride=1, dropout_rate=dropout_rate),
            ResidualBlock(256, 256, stride=1, dropout_rate=dropout_rate),
            ResidualBlock(256, 256, stride=1, dropout_rate=dropout_rate)
        )
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv_block3 = nn.Sequential(
            ResidualBlock(256, 512, stride=1, dropout_rate=dropout_rate),
            ResidualBlock(512, 512, stride=1, dropout_rate=dropout_rate),
            ResidualBlock(512, 512, stride=1, dropout_rate=dropout_rate)
        )

    def forward(self, x):
        x1 = self.conv_block1(x)
        x = self.pool1(x1)
        x2 = self.conv_block2(x)
        x = self.pool2(x2)
        x3 = self.conv_block3(x)
        return x1, x2, x3

# Enhanced BMCNN with HVCs
class EnhancedBMCNNwHVCs(BMCNNBase):
    def __init__(self, num_classes=46, dropout_rate=0.05):
        super(EnhancedBMCNNwHVCs, self).__init__(dropout_rate)
        self.hvc1 = HVCLayer(in_features=128 * 32 * 32, num_classes=num_classes)
        self.hvc2 = HVCLayer(in_features=256 * 16 * 16, num_classes=num_classes)
        self.hvc3 = HVCLayer(in_features=512 * 8 * 8, num_classes=num_classes)
        self.merging = MergingLayer(num_branches=3)

    def forward(self, x):
        x1, x2, x3 = super().forward(x)
        x1_flatten = x1.view(x1.size(0), -1)  # (batch_size, 128*32*32)
        logit1 = self.hvc1(x1_flatten)
        x2_flatten = x2.view(x2.size(0), -1)  # (batch_size, 256*16*16)
        logit2 = self.hvc2(x2_flatten)
        x3_flatten = x3.view(x3.size(0), -1)  # (batch_size, 512*8*8)
        logit3 = self.hvc3(x3_flatten)
        logits = self.merging((logit1, logit2, logit3))
        return logits

# Label smoothing loss
class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes=46, smoothing=0.05):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes

    def forward(self, x, target):
        logprobs = F.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()

# Dataset for Nepali handwritten characters
class NepaliMNISTDataset(Dataset):
    def __init__(self, data_dir, transform=None, logger=None):
        self.data_dir = data_dir
        self.transform = transform
        self.logger = logger
        self.image_paths = []
        self.labels = []
        if not os.path.exists(data_dir):
            raise FileNotFoundError(f"Dataset directory {data_dir} does not exist")
        for class_id in range(46):
            class_dir = os.path.join(data_dir, str(class_id))
            if not os.path.isdir(class_dir):
                if self.logger:
                    self.logger.warning(f"Class directory {class_dir} not found")
                continue
            for img_name in os.listdir(class_dir):
                img_path = os.path.join(class_dir, img_name)
                if not os.path.isfile(img_path):
                    if self.logger:
                        self.logger.warning(f"Image {img_path} is not a file")
                    continue
                self.image_paths.append(img_path)
                self.labels.append(class_id)
        if self.logger:
            self.logger.info(f"Loaded {len(self.image_paths)} images from {data_dir}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        try:
            image = Image.open(img_path).convert('L')
        except Exception as e:
            if self.logger:
                self.logger.error(f"Failed to load image {img_path}: {str(e)}")
            raise
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

# Training function with logging
def train_models(config, train_dataset, val_dataset, device, logger):
    num_models = config.get('num_models', 1)
    seeds = config.get('seeds', [42])
    if len(seeds) != num_models:
        seeds = [42 + i for i in range(num_models)]

    # Initialize epoch log
    epoch_log_file = 'epoch_logs.csv'
    if not os.path.exists(epoch_log_file):
        with open(epoch_log_file, 'w') as f:
            f.write("config_id,seed,epoch,train_loss,train_acc,val_loss,val_acc,epoch_time\n")

    # Results dictionary to store metrics
    config_results = {
        'config': config,
        'models': []
    }

    for model_idx in range(num_models):
        logger.info(f"Training model {model_idx+1}/{num_models} with seed {seeds[model_idx]}")
        # Set seeds for reproducibility
        torch.manual_seed(seeds[model_idx])
        torch.cuda.manual_seed(seeds[model_idx])
        np.random.seed(seeds[model_idx])
        random.seed(seeds[model_idx])

        model = EnhancedBMCNNwHVCs(num_classes=46, dropout_rate=config['dropout']).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config['lr'],
            weight_decay=1e-4
        )

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            min_lr=1e-6
        )

        scaler = GradScaler()

        criterion = LabelSmoothingLoss(classes=46, smoothing=config.get('label_smoothing', 0.05))

        train_loader = DataLoader(
            train_dataset,
            batch_size=config['batch_size'],
            shuffle=True,
            pin_memory=True,
            num_workers=4,
            drop_last=True
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=config['batch_size'],
            shuffle=False,
            pin_memory=True,
            num_workers=4,
            drop_last=True
        )

        best_val_loss = float('inf')
        best_val_acc = 0.0
        patience_counter = 0
        epochs = config.get('epochs', 100)
        patience = config.get('patience', 20)
        best_model_state = None

        # Model-specific metrics
        model_metrics = {
            'seed': seeds[model_idx],
            'epochs': []
        }

        for epoch in range(epochs):
            start_time = time.time()
            model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            for images, labels in train_loader:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                optimizer.zero_grad()
                with autocast('cuda'):
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                running_loss += loss.item() * images.size(0)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
            train_loss = running_loss / len(train_dataset)
            train_acc = correct / total

            model.eval()
            val_loss = 0.0
            correct = 0
            total = 0
            with torch.no_grad():
                for images, labels in val_loader:
                    images = images.to(device, non_blocking=True)
                    labels = labels.to(device, non_blocking=True)
                    with autocast('cuda'):
                        outputs = model(images)
                        loss = criterion(outputs, labels)
                    val_loss += loss.item() * images.size(0)
                    _, predicted = outputs.max(1)
                    total += labels.size(0)
                    correct += predicted.eq(labels).sum().item()
            val_loss /= len(val_dataset)
            val_acc = correct / total

            end_time = time.time()
            epoch_time = end_time - start_time

            # Log metrics
            epoch_metrics = {
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'train_acc': train_acc,
                'val_loss': val_loss,
                'val_acc': val_acc,
                'epoch_time': epoch_time
            }
            model_metrics['epochs'].append(epoch_metrics)

            logger.info(
                f"Config: {config}, Model {model_idx+1}, Epoch {epoch+1}/{epochs} - "
                f"Train Loss: {train_loss:.4f} - Train Acc: {train_acc:.4f} - "
                f"Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.4f} - "
                f"Epoch Time: {epoch_time:.2f}s"
            )
            config_id = f"lr{config['lr']}_bs{config['batch_size']}_dr{config['dropout']}_ls{config['label_smoothing']}"
            with open('epoch_logs.csv', 'a') as f:
                f.write(f"{config_id},{seeds[model_idx]},{epoch+1},{train_loss:.4f},{train_acc:.4f},{val_loss:.4f},{val_acc:.4f},{epoch_time:.2f}\n")
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_state = model.state_dict()
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(f"Model {model_idx+1} Early stopping triggered")
                    break
            scheduler.step(val_loss)

        config_results['models'].append(model_metrics)

    return config_results, best_val_acc, best_model_state

# Evaluation function with detailed metrics
def evaluate_model(model, test_loader, device, criterion, logger, num_classes=46):
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    test_loss /= len(test_loader.dataset)
    test_acc = correct / total

    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average=None, labels=range(num_classes), zero_division=0
    )
    conf_matrix = confusion_matrix(all_labels, all_preds, labels=range(num_classes))

    logger.info(f"Test Loss: {test_loss:.4f} - Test Acc: {test_acc:.4f}")
    return test_loss, test_acc, precision, recall, f1, conf_matrix

# Main function
def main():
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    logger = setup_logging()
    logger.info(f"Starting experiment on {device}")
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}, Device count: {torch.cuda.device_count()}")

    # Data augmentation
    train_transform = transforms.Compose([
        transforms.RandomRotation(15),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
        AddGaussianNoise(mean=0., std=0.03)
    ])
    val_test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # Update data_dir for Google Colab or local environment
    # For Colab, upload dataset or mount Google Drive
    # Example: data_dir = '/content/drive/MyDrive/nepali-quantum-mnist/dhcd'
    train_dataset = NepaliMNISTDataset(
        data_dir='/home/sahaj/nepali-quantum-mnist/dhcd/train',
        transform=train_transform,
        logger=logger
    )
    test_dataset = NepaliMNISTDataset(
        data_dir='/home/sahaj/nepali-quantum-mnist/dhcd/test',
        transform=val_test_transform,
        logger=logger
    )

    train_indices, val_indices = train_test_split(
        list(range(len(train_dataset))),
        test_size=0.2,
        random_state=42,
        stratify=train_dataset.labels
    )
    train_subset = Subset(train_dataset, train_indices)
    val_subset = Subset(train_dataset, val_indices)

    # Fixed parameters
    fixed_params = {
        'model_type': 'hvc',
        'num_models': 1,
        'seeds': [42],
        'epochs': 100,
        'patience': 20
    }

    # Hyperparameters
    hyperparams = [
        {'lr': lr, 'batch_size': bs, 'dropout': dr, 'label_smoothing': ls}
        for lr in [1e-3, 5e-4]
        for bs in [64, 128]
        for dr in [0.0, 0.1]
        for ls in [0.0, 0.1]
    ]

    hyperparam_results = []
    best_val_acc = 0.0
    best_config = None
    best_model_state = None

    for hp in hyperparams:
        config = fixed_params.copy()
        config.update(hp)
        config_results, val_acc, model_state = train_models(config, train_subset, val_subset, device, logger)

        # Aggregate best metrics
        best_train_loss = float('inf')
        best_train_acc = 0.0
        best_val_loss = float('inf')
        best_val_acc_config = 0.0
        for model in config_results['models']:
            for epoch_data in model['epochs']:
                best_train_loss = min(best_train_loss, epoch_data['train_loss'])
                best_train_acc = max(best_train_acc, epoch_data['train_acc'])
                best_val_loss = min(best_val_loss, epoch_data['val_loss'])
                best_val_acc_config = max(best_val_acc_config, epoch_data['val_acc'])

        result_entry = {
            'lr': hp['lr'],
            'batch_size': hp['batch_size'],
            'dropout': hp['dropout'],
            'label_smoothing': hp['label_smoothing'],
            'best_train_loss': best_train_loss,
            'best_train_acc': best_train_acc,
            'best_val_loss': best_val_loss,
            'best_val_acc': best_val_acc_config
        }
        hyperparam_results.append(result_entry)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_config = config.copy()
            best_model_state = model_state

        pd.DataFrame(hyperparam_results).to_csv('hyperparam_results.csv', index=False)
        logger.info(f"Saved hyperparam results for config: {config}")
        logger.info(f"Best metrics for config {config}: {result_entry}")

    # Save best model and evaluate
    if best_config is None or best_model_state is None:
        raise ValueError("No valid best config or model state found")
    logger.info(f"Saving best model with config: {best_config}")
    model = EnhancedBMCNNwHVCs(num_classes=46, dropout_rate=best_config['dropout']).to(device)
    model.load_state_dict(best_model_state)
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': best_config,
        'val_acc': best_val_acc
    }, 'best_model.pth')

    test_loader = DataLoader(test_dataset, batch_size=best_config['batch_size'], shuffle=False, num_workers=4)
    criterion = LabelSmoothingLoss(classes=46, smoothing=best_config['label_smoothing'])
    test_loss, test_acc, precision, recall, f1, conf_matrix = evaluate_model(
        model, test_loader, device, criterion, logger
    )
    metrics_summary = {
        'test_loss': test_loss,
        'test_acc': test_acc,
        'precision': precision.tolist(),
        'recall': recall.tolist(),
        'f1_score': f1.tolist(),
        'conf_matrix': conf_matrix.tolist()
    }
    with open('test_metrics.csv', 'w') as f:
        f.write("metric,value\n")
        f.write(f"test_loss,{test_loss:.4f}\n")
        f.write(f"test_acc,{test_acc:.4f}\n")
        for i in range(46):
            f.write(f"precision_class_{i},{precision[i]:.4f}\n")
            f.write(f"recall_class_{i},{recall[i]:.4f}\n")
            f.write(f"f1_class_{i},{f1[i]:.4f}\n")
    logger.info(f"Test Loss: {test_loss:.4f} - Test Accuracy: {test_acc:.4f}")

if __name__ == '__main__':
    main()