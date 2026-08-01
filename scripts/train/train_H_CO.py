#!/usr/bin/env python3
"""
DiatomScanNet L2: Class  Order Training
- Two-level hierarchical model
- Builds on pretrained class-level model
- Applies hierarchical constraints
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

# Add project root to path for utils
from diatom_cascade.prediction import greedy_hierarchical_predict as hierarchical_predict
from diatom_cascade.training import FocalLoss, masked_focal_loss
from diatom_cascade.data.integrity import load_split_manifests, validate_images
from diatom_cascade.config.train_and_val_config import TrainAndValConfig as TrainingConfig
from diatom_cascade.config.path_config import get_data_root, get_output_dir
from diatom_cascade.config.reporting import print_training_config
from diatom_cascade.checkpoints import (
    CHECKPOINT_SCHEMA_VERSION,
    load_backbone_from_checkpoint,
)
from diatom_cascade.models import ClassToOrderModel

# Config
class Config:
    # Model type (used to load correct dataset from mapping)
    MODEL_TYPE = "H-CO"
    
    # Data paths
    DATA_ROOT = get_data_root()
    IMAGES_DIR = DATA_ROOT / "raw" / "images"
    LABELS_CSV = None  # Will be loaded from model_data_mapping.json
    
    # Prediction method (used in validation)
    PREDICTION_METHOD = "greedy_hierarchical_predict"  # From utils.predict
    
    # Model config
    MODEL_NAME = TrainingConfig.BASE_MODEL
    IMAGE_SIZE = TrainingConfig.IMAGE_SIZE
    NUM_CLASSES = 0  
    NUM_ORDERS = 0   
    
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
    
    # Pretrained model
    CLASS_MODEL_PATH = CHECKPOINT_DIR / "best_F_C_model.pth"

# Create output dirs
Config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
Config.CHECKPOINT_DIR.mkdir(exist_ok=True)
Config.LOG_DIR.mkdir(exist_ok=True)


class ClassToOrderDataset(Dataset):
    """"""
    def __init__(self, df, class_encoder, order_encoder, transform=None):
        self.df = df.reset_index(drop=True)
        self.class_encoder = class_encoder
        self.order_encoder = order_encoder
        self.transform = transform
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_path = Config.IMAGES_DIR / row['filename']
        
        # 
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        
        # 
        class_label = self.class_encoder.transform([row['class']])[0]
        order_label = self.order_encoder.transform([row['order']])[0]
        
        return image, class_label, order_label

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
    
    class_counts = df['class'].value_counts()
    order_counts = df['order'].value_counts()
    
    # Encode labels
    class_encoder = LabelEncoder()
    class_encoder.fit(df['class'].unique())
    
    order_encoder = LabelEncoder()
    order_encoder.fit(df['order'].unique())
    
    Config.NUM_CLASSES = len(class_encoder.classes_)
    Config.NUM_ORDERS = len(order_encoder.classes_)
    
    # Build hierarchy mask from filtered dataframe
    from diatom_cascade.data.hierarchy import build_hierarchy_masks_from_dataframe
    masks = build_hierarchy_masks_from_dataframe(
        df, class_encoder, order_encoder
    )
    
    Config.M_CLASS_ORDER = masks['M_class_order']
    
    return df, class_encoder, order_encoder

def create_data_loaders(df, class_encoder, order_encoder):
    """"""
    # 
    # Use standardized transforms
    train_transforms = TrainingConfig.get_train_transforms()
    val_transforms = TrainingConfig.get_val_test_transforms()
    
    train_df, val_df, test_df = load_split_manifests(Config.DATA_ROOT, Config.MODEL_TYPE)
    validate_images(pd.concat([train_df, val_df]), Config.IMAGES_DIR,
                    Config.OUTPUT_DIR / "preflight" / f"{Config.MODEL_TYPE}_train_validation_images.csv")
    
    
    # 
    train_dataset = ClassToOrderDataset(train_df, class_encoder, order_encoder, train_transforms)
    val_dataset = ClassToOrderDataset(val_df, class_encoder, order_encoder, val_transforms)
    test_dataset = ClassToOrderDataset(test_df, class_encoder, order_encoder, val_transforms)
    
    # 
    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True, 
                             num_workers=TrainingConfig.NUM_WORKERS, pin_memory=TrainingConfig.PIN_MEMORY)
    val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, 
                           num_workers=TrainingConfig.NUM_WORKERS, pin_memory=TrainingConfig.PIN_MEMORY)
    test_loader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, 
                            num_workers=TrainingConfig.NUM_WORKERS, pin_memory=TrainingConfig.PIN_MEMORY)
    
    return train_loader, val_loader, test_loader

# Using unified loss functions from utils.train_and_val

def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train one epoch"""
    model.train()
    total_loss = 0
    class_loss_total = 0
    order_loss_total = 0
    class_correct = 0
    order_correct = 0
    total_samples = 0
    
    for images, class_labels, order_labels in train_loader:
        images = images.to(device)
        class_labels = class_labels.to(device)
        order_labels = order_labels.to(device)
        
        optimizer.zero_grad()
        class_logits, order_logits, class_probs = model(images)
        
        # Loss calculation (Hierarchical Focal Loss)
        class_loss = criterion(class_logits, class_labels)
        
        # Masked Focal Loss for order level
        m = Config.M_CLASS_ORDER.to(device)[class_labels]
        order_loss = masked_focal_loss(order_logits, order_labels, m, alpha=TrainingConfig.FOCAL_ALPHA, gamma=TrainingConfig.FOCAL_GAMMA)
        
        # Get loss weights from config
        weights = TrainingConfig.LOSS_WEIGHTS["H-CO"]
        total_loss_batch = weights["class"] * class_loss + weights["order"] * order_loss
        
        total_loss_batch.backward()
        optimizer.step()
        
        # Statistics
        total_loss += total_loss_batch.item()
        class_loss_total += class_loss.item()
        order_loss_total += order_loss.item()
        
        # Use greedy hierarchical prediction (with mask constraints) for consistent accuracy calculation
        with torch.no_grad():
            pred_class, pred_order = hierarchical_predict(
                class_logits.detach(), order_logits.detach(),
                M_class_order=Config.M_CLASS_ORDER
            )
        
        class_correct += (pred_class == class_labels).sum().item()
        order_correct += (pred_order == order_labels).sum().item()
        total_samples += images.size(0)
    
    num_batches = len(train_loader)
    avg_loss = total_loss / num_batches
    class_acc = class_correct / total_samples
    order_acc = order_correct / total_samples
    avg_class_loss = class_loss_total / num_batches
    avg_order_loss = order_loss_total / num_batches
    
    return avg_loss, class_acc, order_acc, avg_class_loss, avg_order_loss

# topdown_predict is now replaced by hierarchical_predict from utils.hierarchical_predict

def validate_epoch(model, val_loader, criterion, device):
    """Validate one epoch"""
    model.eval()
    total_loss = 0
    class_loss_total = 0
    order_loss_total = 0
    class_correct = 0
    order_correct = 0
    total_samples = 0
    
    # For F1 calculation
    all_class_preds = []
    all_class_labels = []
    all_order_preds = []
    all_order_labels = []
    
    with torch.no_grad():
        for images, class_labels, order_labels in val_loader:
            images = images.to(device)
            class_labels = class_labels.to(device)
            order_labels = order_labels.to(device)
            
            class_logits, order_logits, class_probs = model(images)
            
            # Loss calculation
            class_loss = criterion(class_logits, class_labels)
            m = Config.M_CLASS_ORDER.to(device)[class_labels]
            order_loss = masked_focal_loss(order_logits, order_labels, m, alpha=TrainingConfig.FOCAL_ALPHA, gamma=TrainingConfig.FOCAL_GAMMA)
            
            # Get loss weights from config
            weights = TrainingConfig.LOSS_WEIGHTS["H-CO"]
            total_loss_batch = weights["class"] * class_loss + weights["order"] * order_loss
            total_loss += total_loss_batch.item()
            class_loss_total += class_loss.item()
            order_loss_total += order_loss.item()
            
            # Greedy hierarchical prediction (with mask constraints)
            pred_class, pred_order = hierarchical_predict(
                class_logits, order_logits, 
                M_class_order=Config.M_CLASS_ORDER
            )
            
            class_correct += (pred_class == class_labels).sum().item()
            order_correct += (pred_order == order_labels).sum().item()
            total_samples += images.size(0)
            
            # Collect for F1 calculation
            all_class_preds.extend(pred_class.cpu().numpy())
            all_class_labels.extend(class_labels.cpu().numpy())
            all_order_preds.extend(pred_order.cpu().numpy())
            all_order_labels.extend(order_labels.cpu().numpy())
    
    num_batches = len(val_loader)
    avg_loss = total_loss / num_batches
    class_acc = class_correct / total_samples
    order_acc = order_correct / total_samples
    avg_class_loss = class_loss_total / num_batches
    avg_order_loss = order_loss_total / num_batches
    
    # Calculate weighted F1 scores
    class_f1 = f1_score(all_class_labels, all_class_preds, average='weighted', zero_division=0)
    order_f1 = f1_score(all_order_labels, all_order_preds, average='weighted', zero_division=0)
    
    return avg_loss, class_acc, order_acc, class_f1, order_f1, avg_class_loss, avg_order_loss

# Plotting function moved to analyze/plot_individual_training_curves.py

def main():
    parser = argparse.ArgumentParser(description='Train H-CO hierarchical model')
    
    # Print separator for model identification
    print("\n" + "=" * 80)
    print(f"Training Model: {Config.MODEL_TYPE}")
    print("=" * 80 + "\n")
    TrainingConfig.set_global_seed()
    
    # Load and prepare data
    df, class_encoder, order_encoder = load_and_prepare_data()
    
    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(df, class_encoder, order_encoder)
    
    # Print training configuration
    print_training_config(Config.MODEL_TYPE, Config, TrainingConfig)
    
    # Create model
    model = ClassToOrderModel(
        Config.NUM_CLASSES,
        Config.NUM_ORDERS,
        Config.MODEL_NAME,
        pretrained=False,
    )
    load_backbone_from_checkpoint(
        Config.CLASS_MODEL_PATH,
        model.backbone,
    )
    model = model.to(Config.DEVICE)
    
    # Optimizer and scheduler
    # All models use Focal Loss for fair comparison
    criterion = FocalLoss(alpha=TrainingConfig.FOCAL_ALPHA, gamma=TrainingConfig.FOCAL_GAMMA)
    optimizer = optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY)
    scheduler = TrainingConfig.get_lr_scheduler(optimizer)  # ReduceLROnPlateau
    
    # Training history
    train_losses = []
    val_losses = []
    train_class_losses = []
    val_class_losses = []
    train_order_losses = []
    val_order_losses = []
    train_class_accs = []
    val_class_accs = []
    train_order_accs = []
    val_order_accs = []
    val_class_f1s = []
    val_order_f1s = []
    
    best_val_f1 = 0  # Use weighted F1 for early stopping
    best_epoch = 0
    patience_counter = 0
    
    pbar = tqdm(range(Config.EPOCHS), desc="Training")
    
    for epoch in pbar:
        # Train
        train_loss, train_class_acc, train_order_acc, train_class_loss, train_order_loss = train_epoch(
            model, train_loader, criterion, optimizer, Config.DEVICE)
        
        # Validate
        val_loss, val_class_acc, val_order_acc, val_class_f1, val_order_f1, val_class_loss, val_order_loss = validate_epoch(
            model, val_loader, criterion, Config.DEVICE)
        
        # Record metrics
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_class_losses.append(train_class_loss)
        val_class_losses.append(val_class_loss)
        train_order_losses.append(train_order_loss)
        val_order_losses.append(val_order_loss)
        train_class_accs.append(train_class_acc)
        val_class_accs.append(val_class_acc)
        train_order_accs.append(train_order_acc)
        val_order_accs.append(val_order_acc)
        val_class_f1s.append(val_class_f1)
        val_order_f1s.append(val_order_f1)
        
        # Update learning rate
        scheduler.step(val_loss)  # ReduceLROnPlateau needs val_loss
        
        # Check if best model (using weighted F1 for imbalanced data)
        is_best = val_order_f1 > best_val_f1
        if is_best:
            best_val_f1 = val_order_f1
            best_epoch = epoch + 1
            patience_counter = 0
            
            # Save best model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_order_f1': val_order_f1,
                'val_order_acc': val_order_acc,
                'val_class_f1': val_class_f1,
                'val_class_acc': val_class_acc,
                'class_names': list(class_encoder.classes_),
                'order_names': list(order_encoder.classes_),
                'num_classes': Config.NUM_CLASSES,
                'num_orders': Config.NUM_ORDERS,
                'M_class_order': Config.M_CLASS_ORDER,
                'checkpoint_schema_version': CHECKPOINT_SCHEMA_VERSION
            }, Config.CHECKPOINT_DIR / 'best_H_CO_model.pth')
            
            pbar.set_postfix({'best_f1': f'{val_order_f1:.4f}', 'epoch': epoch+1})
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= Config.PATIENCE:
            break
    
    print(f"\n✅ Training complete: Best Order F1={best_val_f1:.4f} at epoch {best_epoch}/{len(train_losses)}")
    
    # Save training history
    history = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_class_loss': train_class_losses,
        'val_class_loss': val_class_losses,
        'train_order_loss': train_order_losses,
        'val_order_loss': val_order_losses,
        'train_class_acc': train_class_accs,
        'val_class_acc': val_class_accs,
        'train_order_accs': train_order_accs,
        'val_order_accs': val_order_accs,
        'val_class_f1': val_class_f1s,
        'val_order_f1': val_order_f1s,
        'best_val_order_f1': best_val_f1,
        'best_epoch': best_epoch
    }
    
    # Save training history (plotting is done separately in analyze/)
    history_path = Config.LOG_DIR / 'H_CO__training_history.json'
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)

if __name__ == "__main__":
    main()
