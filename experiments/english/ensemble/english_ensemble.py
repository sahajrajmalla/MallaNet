import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import MNIST
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
import os
from torch.amp import autocast, GradScaler
import logging
import sys
import time
import warnings
import random
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
from torchvision.transforms import ElasticTransform, RandomErasing

warnings.filterwarnings("ignore", category=UserWarning)

os.environ["OMP_NUM_THREADS"] = "1"

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

class AddGaussianNoise(object):
    def __init__(self, mean=0., std=0.01):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

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

class HFCLayer(nn.Module):
    def __init__(self, num_classes, D_b):
        super(HFCLayer, self).__init__()
        self.num_classes = num_classes
        self.V = nn.Parameter(torch.randn(num_classes, D_b))
        self.bn = nn.BatchNorm1d(num_classes * D_b)

    def forward(self, x):
        U_b = x.sum(dim=1)
        U_b_exp = U_b.unsqueeze(1)
        V_exp = self.V.unsqueeze(0)
        T_b = U_b_exp * V_exp
        batch_size = T_b.size(0)
        T_b_flat = T_b.view(batch_size, -1)
        T_b_bn = self.bn(T_b_flat)
        T_b_bn = T_b_bn.view(batch_size, self.num_classes, -1)
        T_b_relu = F.relu(T_b_bn)
        logits = T_b_relu.sum(dim=2)
        return logits

class MergingLayer(nn.Module):
    def __init__(self, num_branches=3):
        super(MergingLayer, self).__init__()
        self.w = nn.Parameter(torch.ones(num_branches) / num_branches)

    def forward(self, inputs):
        weights = F.softmax(self.w, dim=0)
        return sum(w * logit for w, logit in zip(weights, inputs))

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

class EnhancedBMCNNwHFCs(BMCNNBase):
    def __init__(self, num_classes=10, dropout_rate=0.05):
        super(EnhancedBMCNNwHFCs, self).__init__(dropout_rate)
        self.hfc1 = HFCLayer(num_classes, D_b=32*32)
        self.hfc2 = HFCLayer(num_classes, D_b=16*16)
        self.hfc3 = HFCLayer(num_classes, D_b=8*8)
        self.merging = MergingLayer(num_branches=3)

    def forward(self, x):
        x1, x2, x3 = super().forward(x)
        x1_reshaped = x1.view(x1.size(0), x1.size(1), -1)
        logit1 = self.hfc1(x1_reshaped)
        x2_reshaped = x2.view(x2.size(0), x2.size(1), -1)
        logit2 = self.hfc2(x2_reshaped)
        x3_reshaped = x3.view(x3.size(0), x3.size(1), -1)
        logit3 = self.hfc3(x3_reshaped)
        logits = self.merging((logit1, logit2, logit3))
        return logits

class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes=10, smoothing=0.05):
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

def tta_transform(image):
    angle = random.uniform(-5, 5)
    image = transforms.functional.rotate(image, angle)
    translate = (random.uniform(-0.02, 0.02), random.uniform(-0.02, 0.02))
    image = transforms.functional.affine(image, angle=0, translate=translate, scale=1.0, shear=0)
    return image

def train_models(config, train_dataset, val_dataset, device, logger):
    num_models = config.get('num_models', 1)
    seeds = config.get('seeds', [42])
    if len(seeds) != num_models:
        seeds = [42 + i for i in range(num_models)]

    epoch_log_file = 'epoch_logs.csv'
    if not os.path.exists(epoch_log_file):
        with open(epoch_log_file, 'w') as f:
            f.write("config_id,seed,epoch,train_loss,train_acc,val_loss,val_acc,epoch_time\n")

    config_results = {
        'config': config,
        'models': []
    }
    best_model_states = []
    best_val_accs = []

    for model_idx in range(num_models):
        logger.info(f"Training model {model_idx+1}/{num_models} with seed {seeds[model_idx]}")
        torch.manual_seed(seeds[model_idx])
        torch.cuda.manual_seed(seeds[model_idx])
        np.random.seed(seeds[model_idx])
        random.seed(seeds[model_idx])

        model = EnhancedBMCNNwHFCs(num_classes=10, dropout_rate=config['dropout']).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config['lr'],
            weight_decay=1e-4
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
        scaler = GradScaler()
        criterion = LabelSmoothingLoss(classes=10, smoothing=config.get('label_smoothing', 0.05))

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
            drop_last=False
        )

        best_val_loss = float('inf')
        best_val_acc = 0.0
        patience_counter = 0
        epochs = config.get('epochs', 100)
        patience = config.get('patience', 20)
        best_model_state = None
        model_metrics = {'epochs': []}

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
                torch.save(best_model_state, f'model_seed_{seeds[model_idx]}.pth')
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(f"Model {model_idx+1} Early stopping triggered")
                    break
            scheduler.step()

        config_results['models'].append(model_metrics)
        best_model_states.append(best_model_state)
        best_val_accs.append(best_val_acc)

    avg_val_acc = sum(best_val_accs) / len(best_val_accs) if best_val_accs else 0.0
    return config_results, best_model_states, avg_val_acc, best_val_accs

def evaluate_model(models, test_loader, device, criterion, logger, num_classes=10, num_tta=5):
    for model in models:
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
            tta_outputs = []
            for _ in range(num_tta):
                tta_images = torch.stack([tta_transform(img) for img in images])
                all_outputs = []
                for model in models:
                    outputs = model(tta_images)
                    all_outputs.append(F.softmax(outputs, dim=1))
                avg_outputs = torch.stack(all_outputs).mean(dim=0)
                tta_outputs.append(avg_outputs)
            final_outputs = torch.stack(tta_outputs).mean(dim=0)
            loss = criterion(final_outputs, labels)
            test_loss += loss.item() * images.size(0)
            _, predicted = final_outputs.max(1)
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

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger = setup_logging()
    logger.info(f"Starting experiment on {device}")
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}, Device count: {torch.cuda.device_count()}")

    mean = 0.1307
    std = 0.3081

    train_transform = transforms.Compose([
        transforms.Resize((32, 32)),
        ElasticTransform(alpha=36.0, sigma=5.0),
        transforms.RandomRotation(10),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.8, 1.2)),
        transforms.ToTensor(),
        RandomErasing(p=0.3, scale=(0.02, 0.2), ratio=(0.3, 3.3), value=0),
        transforms.Normalize((mean,), (std,)),
        AddGaussianNoise(mean=0., std=0.01)
    ])
    val_test_transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((mean,), (std,))
    ])

    train_dataset = MNIST(
        root='data',
        train=True,
        download=True,
        transform=train_transform
    )
    val_dataset = MNIST(
        root='data',
        train=True,
        download=True,
        transform=val_test_transform
    )
    test_dataset = MNIST(
        root='data',
        train=False,
        download=True,
        transform=val_test_transform
    )

    train_indices, val_indices = train_test_split(
        list(range(len(train_dataset))),
        test_size=0.2,
        random_state=42,
        stratify=train_dataset.targets
    )
    train_subset = Subset(train_dataset, train_indices)
    val_subset = Subset(val_dataset, val_indices)

    best_config = {
        'model_type': 'hfc',
        'num_models': 3,
        'seeds': [42, 43, 44],
        'epochs': 100,
        'patience': 20,
        'lr': 0.001,
        'batch_size': 64,
        'dropout': 0.2,
        'label_smoothing': 0.2
    }

    config_results, best_model_states, avg_val_acc, best_val_accs = train_models(best_config, train_subset, val_subset, device, logger)

    models = []
    for idx, model_state in enumerate(best_model_states):
        model = EnhancedBMCNNwHFCs(num_classes=10, dropout_rate=best_config['dropout']).to(device)
        model.load_state_dict(model_state)
        models.append(model)
        torch.save({
            'model_state_dict': model_state,
            'config': best_config,
            'val_acc': best_val_accs[idx]
        }, f'best_model_seed_{best_config["seeds"][idx]}.pth')

    test_loader = DataLoader(test_dataset, batch_size=best_config['batch_size'], shuffle=False, num_workers=4)
    criterion = LabelSmoothingLoss(classes=10, smoothing=best_config['label_smoothing'])
    test_loss, test_acc, precision, recall, f1, conf_matrix = evaluate_model(
        models, test_loader, device, criterion, logger, num_tta=5
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
        for i in range(10):
            f.write(f"precision_class_{i},{precision[i]:.4f}\n")
            f.write(f"recall_class_{i},{recall[i]:.4f}\n")
            f.write(f"f1_class_{i},{f1[i]:.4f}\n")
    logger.info(f"Test Loss: {test_loss:.4f} - Test Accuracy: {test_acc:.4f}")

if __name__ == '__main__':
    main()