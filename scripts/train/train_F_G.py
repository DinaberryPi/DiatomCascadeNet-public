#!/usr/bin/env python3
"""
DiatomScanNet F Genus Baseline Training
- Direct classification of Genus without hierarchical structure
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
from torch.utils.data import Dataset, DataLoader
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
from diatom_cascade.config.reporting import print_training_config
from diatom_cascade.models import FlatClassifier
from diatom_cascade.checkpoints import CHECKPOINT_SCHEMA_VERSION

# Config
class Config:
    # Model type (used to load correct dataset from mapping)
    MODEL_TYPE = "F-G"
    
    # Data paths
    DATA_ROOT = get_data_root()
    IMAGES_DIR = DATA_ROOT / "raw" / "images"
    LABELS_CSV = None  # Will be loaded from model_data_mapping.json
    
    # Prediction method (used in validation)
    PREDICTION_METHOD = "torch.argmax"  # Direct argmax for flat models
    
    # Model config
    MODEL_NAME = TrainingConfig.BASE_MODEL
    IMAGE_SIZE = TrainingConfig.IMAGE_SIZE
    NUM_CLASSES = 0      # Genus (flat classification)
    
    # Training (from standardized config)
    BATCH_SIZE = TrainingConfig.BATCH_SIZE
    EPOCHS = TrainingConfig.MAX_EPOCHS
    LEARNING_RATE = TrainingConfig.INITIAL_LR
    WEIGHT_DECAY = TrainingConfig.WEIGHT_DECAY
    PATIENCE = TrainingConfig.EARLY_STOPPING_PATIENCE
    
    # Device
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Output
    OUTPUT_DIR = get_output_dir()
    CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
    LOG_DIR = OUTPUT_DIR / "logs"
    
# Create output dirs
Config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
Config.CHECKPOINT_DIR.mkdir(exist_ok=True)
Config.LOG_DIR.mkdir(exist_ok=True)


# Import unified loss functions
from diatom_cascade.training import FocalLoss
from diatom_cascade.data.integrity import load_split_manifests, validate_images

class FGDataset(Dataset):
    """Dataset for F-G model (flat genus classification)"""
    def __init__(self, df, genus_encoder, transform=None):
        self.df = df.reset_index(drop=True)
        self.genus_encoder = genus_encoder
        self.transform = transform
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_path = Config.IMAGES_DIR / row['filename']
        
        # Load image
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        
        # Encode labels (only genus)
        genus_label = int(self.genus_encoder.transform([row['genus']])[0])
        
        return image, genus_label

def load_and_prepare_data():
    """Load and preprocess data"""
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
    
    # Encode labels (only genus)
    genus_encoder = LabelEncoder()
    genus_encoder.fit(df['genus'].unique())
    
    Config.NUM_CLASSES = len(genus_encoder.classes_)
    
    return df, genus_encoder

def create_data_loaders(df, genus_encoder):
    """Create data loaders"""
    # Use standardized transforms
    train_transforms = TrainingConfig.get_train_transforms()
    val_transforms = TrainingConfig.get_val_test_transforms()
    
    train_df, val_df, test_df = load_split_manifests(Config.DATA_ROOT, Config.MODEL_TYPE)
    validate_images(pd.concat([train_df, val_df]), Config.IMAGES_DIR,
                    Config.OUTPUT_DIR / "preflight" / f"{Config.MODEL_TYPE}_train_validation_images.csv")
    
    train_dataset = FGDataset(train_df, genus_encoder, train_transforms)
    val_dataset = FGDataset(val_df, genus_encoder, val_transforms)
    test_dataset = FGDataset(test_df, genus_encoder, val_transforms)
    
    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True, 
                             num_workers=TrainingConfig.NUM_WORKERS, pin_memory=TrainingConfig.PIN_MEMORY)
    val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, 
                           num_workers=TrainingConfig.NUM_WORKERS, pin_memory=TrainingConfig.PIN_MEMORY)
    test_loader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, 
                            num_workers=TrainingConfig.NUM_WORKERS, pin_memory=TrainingConfig.PIN_MEMORY)
    
    return train_loader, val_loader, test_loader

def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train one epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total_samples = 0
    
    for images, genus_labels in train_loader:
        images = images.to(device)
        genus_labels = genus_labels.to(device)
        
        optimizer.zero_grad()
        logits = model(images)
        
        # Single loss for genus classification
        loss = criterion(logits, genus_labels)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        # Calculate accuracy
        with torch.no_grad():
            _, predicted = torch.max(logits, 1)
            correct += (predicted == genus_labels).sum().item()
        
        total_samples += images.size(0)
    
    num_batches = len(train_loader)
    avg_loss = total_loss / num_batches
    acc = correct / total_samples
    
    return avg_loss, acc

def validate_epoch(model, val_loader, criterion, device):
    """Validate one epoch"""
    model.eval()
    total_loss = 0
    correct = 0
    total_samples = 0
    
    # For F1 calculation
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, genus_labels in val_loader:
            images = images.to(device)
            genus_labels = genus_labels.to(device)
            
            logits = model(images)
            
            # Loss
            loss = criterion(logits, genus_labels)
            total_loss += loss.item()
            
            # Predictions
            _, predicted = torch.max(logits, 1)
            correct += (predicted == genus_labels).sum().item()
            total_samples += images.size(0)
            
            # Collect for F1 calculation
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(genus_labels.cpu().numpy())
    
    num_batches = len(val_loader)
    avg_loss = total_loss / num_batches
    acc = correct / total_samples
    
    # Calculate weighted F1 score
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    
    return avg_loss, acc, f1

def main():
    parser = argparse.ArgumentParser(description='Train F-G flat model')
    
    # Print separator for model identification
    print("\n" + "=" * 80)
    print(f"Training Model: {Config.MODEL_TYPE}")
    print("=" * 80 + "\n")
    TrainingConfig.set_global_seed()
    
    # Load and prepare data
    df, genus_encoder = load_and_prepare_data()
    
    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(
        df, genus_encoder
    )
    
    # Print training configuration
    print_training_config(Config.MODEL_TYPE, Config, TrainingConfig)
    
    # Create model
    model = FlatClassifier(
        Config.NUM_CLASSES,
        Config.MODEL_NAME,
        pretrained=TrainingConfig.BACKBONE_PRETRAIN == "imagenet",
    )
    model = model.to(Config.DEVICE)
    
    # Optimizer and scheduler (from standardized config)
    # F-G uses Cross-Entropy Loss as a simple baseline for error propagation analysis
    # All models use Focal Loss for fair comparison
    criterion = FocalLoss(alpha=TrainingConfig.FOCAL_ALPHA, gamma=TrainingConfig.FOCAL_GAMMA)
    optimizer = optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY)
    scheduler = TrainingConfig.get_lr_scheduler(optimizer)  # ReduceLROnPlateau
    
    # Training history
    # For flat classifier: save as genus metrics (no family metrics)
    history = {
        'train_loss': [], 'val_loss': [],  # Combined loss (same as genus for flat classifier)
        'train_genus_loss': [], 'val_genus_loss': [],
        'train_genus_acc': [], 'val_genus_acc': [],
        'val_genus_f1': []
    }
    
    best_val_f1 = 0
    best_epoch = 0
    patience_counter = 0
    
    pbar = tqdm(range(Config.EPOCHS), desc="Training")
    
    for epoch in pbar:
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, Config.DEVICE
        )
        
        # Validate
        val_loss, val_acc, val_f1 = validate_epoch(
            model, val_loader, criterion, Config.DEVICE
        )
        
        # Update history
        # For flat classifier: train_loss = genus_loss (no family component)
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_genus_loss'].append(train_loss)
        history['val_genus_loss'].append(val_loss)
        history['train_genus_acc'].append(train_acc)
        history['val_genus_acc'].append(val_acc)
        history['val_genus_f1'].append(val_f1)
        
        # Update learning rate (ReduceLROnPlateau needs val_loss)
        scheduler.step(val_loss)
        
        # Check if best model
        is_best = val_f1 > best_val_f1
        if is_best:
            best_val_f1 = val_f1
            best_epoch = epoch + 1
            patience_counter = 0
            
            # Save best model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_f1': val_f1,
                'val_acc': val_acc,
                'genus_names': list(genus_encoder.classes_),
                'num_classes': Config.NUM_CLASSES,
                'config': {
                    'NUM_CLASSES': Config.NUM_CLASSES,
                    'MODEL_NAME': Config.MODEL_NAME
                },
                'checkpoint_schema_version': CHECKPOINT_SCHEMA_VERSION
            }, Config.CHECKPOINT_DIR / 'best_F_G_model.pth')
            
            pbar.set_postfix({'best_f1': f'{val_f1:.4f}', 'epoch': epoch+1})
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= Config.PATIENCE:
            break
    
    print(f"\n✅ Training complete: Best F1={best_val_f1:.4f} at epoch {best_epoch}/{len(history['train_loss'])}")
    
    # Save training history
    history['best_val_f1'] = best_val_f1
    history['best_epoch'] = best_epoch
    
    history_path = Config.LOG_DIR / 'F_G__training_history.json'
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)

if __name__ == "__main__":
    main()
