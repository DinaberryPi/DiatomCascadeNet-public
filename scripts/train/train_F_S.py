#!/usr/bin/env python3
"""
DiatomScanNet F-S: Flat Species Baseline Training
- EfficientNet-B0 backbone
- Single-head classification (Species level only)
- Flat baseline model for hierarchical comparison
"""

import os
import sys
import json
import random
import argparse
from pathlib import Path
from collections import Counter, defaultdict
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.transforms as transforms
from PIL import Image
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, f1_score, accuracy_score
from sklearn.preprocessing import LabelEncoder
import timm
from tqdm import tqdm
import sys
from pathlib import Path

# Import standardized training config
from diatom_cascade.config.train_and_val_config import TrainAndValConfig as TrainingConfig
from diatom_cascade.config.path_config import get_data_root, get_output_dir
from diatom_cascade.training import FocalLoss
from diatom_cascade.data.integrity import load_split_manifests, validate_images
from diatom_cascade.config.reporting import print_training_config
from diatom_cascade.models import FlatClassifier as EfficientNetClassifier
from diatom_cascade.checkpoints import CHECKPOINT_SCHEMA_VERSION

class Config:
    # Model type (used to load correct dataset from mapping)
    MODEL_TYPE = "F-S"
    
    # Data
    DATA_ROOT = get_data_root()
    IMAGES_DIR = DATA_ROOT / "raw" / "images"
    LABELS_CSV = None  # Will be loaded from model_data_mapping.json
    
    # Prediction method (used in validation)
    PREDICTION_METHOD = "torch.argmax"  # Direct argmax for flat models
    
    # Model
    MODEL_NAME = TrainingConfig.BASE_MODEL
    IMAGE_SIZE = TrainingConfig.IMAGE_SIZE
    NUM_SPECIES = 0
    
    # Training (from standardized config)
    BATCH_SIZE = TrainingConfig.BATCH_SIZE
    EPOCHS = TrainingConfig.MAX_EPOCHS
    LEARNING_RATE = TrainingConfig.INITIAL_LR
    WEIGHT_DECAY = TrainingConfig.WEIGHT_DECAY
    FOCAL_ALPHA = TrainingConfig.FOCAL_ALPHA
    FOCAL_GAMMA = TrainingConfig.FOCAL_GAMMA
    
    # Data filtering
    MIN_SAMPLES = TrainingConfig.MIN_SAMPLES["Species"]  # Minimum samples per species
    
    # Early stopping (from standardized config)
    EARLY_STOP_PATIENCE = TrainingConfig.EARLY_STOPPING_PATIENCE
    MIN_DELTA = TrainingConfig.EARLY_STOPPING_MIN_DELTA
    
    # Device
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Output
    OUTPUT_DIR = get_output_dir()
    CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
    LOG_DIR = OUTPUT_DIR / "logs"

# Create output directories
Config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
Config.CHECKPOINT_DIR.mkdir(exist_ok=True)
Config.LOG_DIR.mkdir(exist_ok=True)

class DiatomDataset(Dataset):
    def __init__(self, df, label_encoder, transforms=None):
        self.df = df.reset_index(drop=True)
        self.transforms = transforms
        self.label_encoder = label_encoder
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = Config.IMAGES_DIR / row['filename']
        
        # Load image
        with Image.open(img_path) as source:
            image = source.convert('RGB')
        
        if self.transforms:
            image = self.transforms(image)
        
        # Encode label
        label = str(row['species'])
        label_id = self.label_encoder.transform([label])[0]
        
        return image, label_id


# Using unified FocalLoss from utils.train_and_val

def get_class_weights_and_sampler(df, target_col='species'):
    """Calculate class weights and sampler"""
    class_counts = df[target_col].value_counts()
    class_weights = 1.0 / np.sqrt(class_counts.values)
    class_weights = class_weights / class_weights.sum() * len(class_weights)
    
    # Create weight mapping
    class_to_idx = {cls: idx for idx, cls in enumerate(class_counts.index)}
    sample_weights = [class_weights[class_to_idx[cls]] for cls in df[target_col]]
    
    # Create sampler
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(df),
        replacement=True
    )
    
    return class_weights, sampler, class_to_idx

def prepare_data():
    """Prepare dataset and dataloaders"""
    # Load model-specific dataset from mapping
    mapping_file = Config.DATA_ROOT / "preprocessed" / "model_data_mapping.json"
    if not mapping_file.exists():
        raise FileNotFoundError(
            f"Model mapping file not found: {mapping_file}\n"
            "Please run: python -m scripts.data.preprocessing.create_filtered_datasets"
        )
    
    with open(mapping_file, 'r', encoding='utf-8') as f:
        model_data_mapping = json.load(f)
    
    if Config.MODEL_TYPE not in model_data_mapping:
        raise ValueError(f"Model type '{Config.MODEL_TYPE}' not found in mapping")
    
    # Load the correct dataset file
    # Handle relative paths (e.g., "../cleaned/labels_clean.csv" for F-C)
    dataset_path = model_data_mapping[Config.MODEL_TYPE]
    dataset_file = (Config.DATA_ROOT / "preprocessed" / dataset_path).resolve()
    Config.LABELS_CSV = dataset_file
    
    # Always show dataset info (even in quiet mode)
    print(f"Model: {Config.MODEL_TYPE}")
    print(f"Dataset: {dataset_file.name}")
    
    df = pd.read_csv(Config.LABELS_CSV)
    
    # Create label encoder
    df['species'] = df['species'].fillna('unknown').astype(str)
    label_encoder = LabelEncoder()
    label_encoder.fit(df['species'].unique())
    
    Config.NUM_SPECIES = len(label_encoder.classes_)
    
    train_df, val_df, test_df = load_split_manifests(Config.DATA_ROOT, Config.MODEL_TYPE)
    validate_images(pd.concat([train_df, val_df]), Config.IMAGES_DIR,
                    Config.OUTPUT_DIR / "preflight" / f"{Config.MODEL_TYPE}_train_validation_images.csv")
    
    # Data augmentation (from standardized config)
    train_transforms = TrainingConfig.get_train_transforms()
    val_transforms = TrainingConfig.get_val_test_transforms()
    
    # Create datasets
    train_dataset = DiatomDataset(train_df, label_encoder, train_transforms)
    val_dataset = DiatomDataset(val_df, label_encoder, val_transforms)
    
    # Create sampler
    _, train_sampler, _ = get_class_weights_and_sampler(train_df, 'species')
    
    # Create dataloaders (from standardized config)
    train_loader = DataLoader(
        train_dataset, 
        batch_size=Config.BATCH_SIZE, 
        sampler=train_sampler,
        num_workers=TrainingConfig.NUM_WORKERS,
        pin_memory=TrainingConfig.PIN_MEMORY
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=Config.BATCH_SIZE, 
        shuffle=False,
        num_workers=TrainingConfig.NUM_WORKERS,
        pin_memory=TrainingConfig.PIN_MEMORY
    )
    
    return train_loader, val_loader, train_df, val_df, test_df, label_encoder

def train_epoch(model, train_loader, optimizer, criterion, device):
    """Train one epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Statistics
        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    return total_loss / len(train_loader), 100. * correct / total

def validate(model, val_loader, criterion, device):
    """Validate"""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            
            # Collect predictions
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate metrics
    f1_macro = f1_score(all_labels, all_preds, average='macro')
    f1_weighted = f1_score(all_labels, all_preds, average='weighted')
    acc = accuracy_score(all_labels, all_preds)
    
    return total_loss / len(val_loader), f1_weighted, acc, f1_macro

class EarlyStopping:
    """Early stopping"""
    def __init__(self, patience=10, min_delta=0.001, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_score = None
        self.counter = 0
        self.best_weights = None
        
    def __call__(self, val_score, model):
        if self.best_score is None:
            self.best_score = val_score
            self.save_checkpoint(model)
        elif val_score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                if self.restore_best_weights:
                    model.load_state_dict(self.best_weights)
                return True
        else:
            self.best_score = val_score
            self.counter = 0
            self.save_checkpoint(model)
        return False
    
    def save_checkpoint(self, model):
        self.best_weights = model.state_dict().copy()

def main():
    parser = argparse.ArgumentParser(description='Train F-S flat model')
    
    # Print separator for model identification
    print("\n" + "=" * 80)
    print(f"Training Model: {Config.MODEL_TYPE}")
    print("=" * 80 + "\n")
    TrainingConfig.set_global_seed()
    
    # Prepare data
    train_loader, val_loader, train_df, val_df, test_df, label_encoder = prepare_data()
    
    # Print training configuration
    print_training_config(Config.MODEL_TYPE, Config, TrainingConfig)
    
    # Create model
    model = EfficientNetClassifier(
        Config.NUM_SPECIES,
        Config.MODEL_NAME,
        pretrained=TrainingConfig.BACKBONE_PRETRAIN == "imagenet",
    )
    model = model.to(Config.DEVICE)
    
    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=Config.LEARNING_RATE, 
        weight_decay=Config.WEIGHT_DECAY
    )
    
    # Learning rate scheduler (from standardized config - ReduceLROnPlateau)
    scheduler = TrainingConfig.get_lr_scheduler(optimizer)
    
    # Loss function
    # All models use Focal Loss for fair comparison
    criterion = FocalLoss(alpha=TrainingConfig.FOCAL_ALPHA, gamma=TrainingConfig.FOCAL_GAMMA)
    
    # Training loop
    best_val_f1 = 0
    train_losses = []
    val_losses = []
    train_accs = []
    val_f1s = []
    val_accs = []
    
    # Early stopping
    early_stopping = EarlyStopping(
        patience=Config.EARLY_STOP_PATIENCE,
        min_delta=Config.MIN_DELTA
    )
    
    pbar = tqdm(range(Config.EPOCHS), desc="Training")
    
    for epoch in pbar:
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, Config.DEVICE)
        
        # Validate
        val_loss, val_f1_weighted, val_acc, val_f1_macro = validate(model, val_loader, criterion, Config.DEVICE)
        
        # Update learning rate (ReduceLROnPlateau needs val_loss)
        scheduler.step(val_loss)
        
        # Record metrics
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_f1s.append(val_f1_weighted)
        val_accs.append(val_acc)
        
        # Check if best model
        is_best = val_f1_weighted > best_val_f1
        if is_best:
            best_val_f1 = val_f1_weighted
            # Save best model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_f1_weighted': val_f1_weighted,
                'val_f1_macro': val_f1_macro,
                'val_acc': val_acc,
                'species_names': list(label_encoder.classes_),
                'num_species': Config.NUM_SPECIES,
                'checkpoint_schema_version': CHECKPOINT_SCHEMA_VERSION
            }, Config.CHECKPOINT_DIR / 'best_F_S_model.pth')
        
        # Print status
        if is_best:
            pbar.set_postfix({'best_f1': f'{val_f1_weighted:.4f}', 'epoch': epoch+1})
        
        # Early stopping check (use same metric as best model selection for consistency)
        if early_stopping(val_f1_weighted, model):
            break
    
    print(f"\n✅ Training complete: Best F1={best_val_f1:.4f} at epoch {len(train_losses)}/{Config.EPOCHS}")
    
    # Save training history
    history = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accs': train_accs,
        'val_f1s': val_f1s,
        'val_accs': val_accs,
        'best_val_f1': best_val_f1
    }
    
    # Save training history (plotting is done separately in analyze/)
    history_path = Config.LOG_DIR / 'F_S__training_history.json'
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)

if __name__ == "__main__":
    main()
