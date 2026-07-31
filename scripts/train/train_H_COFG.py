#!/usr/bin/env python3
"""
DiatomScanNet H-COFG Training
Level 4: Class → Order → Family → Genus
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

# Add project root to path for utils

from diatom_cascade.prediction import greedy_hierarchical_predict as hierarchical_predict
from diatom_cascade.training import FocalLoss, masked_focal_loss
from diatom_cascade.data.integrity import load_split_manifests, validate_images
from diatom_cascade.config.train_and_val_config import TrainAndValConfig as TrainingConfig
from diatom_cascade.config.path_config import get_output_dir
from diatom_cascade.config.reporting import print_training_config
from diatom_cascade.checkpoints import (
    CHECKPOINT_SCHEMA_VERSION,
    load_backbone_from_checkpoint,
)
from diatom_cascade.models import HCOFGModel


# Config
class Config:
    # Model type (used to load correct dataset from mapping)
    MODEL_TYPE = "H-COFG"
    
    # Data paths
    DATA_ROOT = Path("dataset")
    IMAGES_DIR = DATA_ROOT / "raw" / "images"
    LABELS_CSV = None  # Will be loaded from model_data_mapping.json
    
    # Prediction method (used in validation)
    PREDICTION_METHOD = "greedy_hierarchical_predict"  # From utils.predict
    
    # Model config
    MODEL_NAME = TrainingConfig.BASE_MODEL
    IMAGE_SIZE = TrainingConfig.IMAGE_SIZE
    NUM_CLASSES = 0     # Class
    NUM_ORDERS = 0      # Order
    NUM_FAMILIES = 0    # Family
    NUM_GENERA = 0      # Genus
    
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
    
    # Pretrained model (use H-COF hierarchical model)
    PRETRAINED_MODEL_PATH = CHECKPOINT_DIR / "best_H_COF_model.pth"

# Create output dirs
Config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
Config.CHECKPOINT_DIR.mkdir(exist_ok=True)
Config.LOG_DIR.mkdir(exist_ok=True)


class HCOFGDataset(Dataset):
    """Dataset for H-COFG model"""
    def __init__(self, df, class_encoder, order_encoder, family_encoder, genus_encoder, transform=None):
        self.df = df.reset_index(drop=True)
        self.class_encoder = class_encoder
        self.order_encoder = order_encoder
        self.family_encoder = family_encoder
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
        
        # Encode labels
        class_label = int(self.class_encoder.transform([row['class']])[0])
        order_label = int(self.order_encoder.transform([row['order']])[0])
        family_label = int(self.family_encoder.transform([row['family']])[0])
        genus_label = int(self.genus_encoder.transform([row['genus']])[0])
        
        return image, class_label, order_label, family_label, genus_label

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
    
    # Always show dataset info
    print(f"Model: {Config.MODEL_TYPE}")
    print(f"Dataset: {dataset_file.name}")
    
    df = pd.read_csv(Config.LABELS_CSV)
    
    class_counts = df['class'].value_counts()
    order_counts = df['order'].value_counts()
    family_counts = df['family'].value_counts()
    genus_counts = df['genus'].value_counts()
    
    # Encode labels
    class_encoder = LabelEncoder()
    class_encoder.fit(df['class'].unique())
    
    order_encoder = LabelEncoder()
    order_encoder.fit(df['order'].unique())
    
    family_encoder = LabelEncoder()
    family_encoder.fit(df['family'].unique())
    
    genus_encoder = LabelEncoder()
    genus_encoder.fit(df['genus'].unique())
    
    Config.NUM_CLASSES = len(class_encoder.classes_)
    Config.NUM_ORDERS = len(order_encoder.classes_)
    Config.NUM_FAMILIES = len(family_encoder.classes_)
    Config.NUM_GENERA = len(genus_encoder.classes_)
    
    # Build hierarchy masks from filtered dataframe
    from diatom_cascade.data.hierarchy import build_hierarchy_masks_from_dataframe
    masks = build_hierarchy_masks_from_dataframe(
        df, class_encoder, order_encoder, family_encoder, genus_encoder
    )
    
    Config.M_CLASS_ORDER = masks['M_class_order']
    Config.M_ORDER_FAMILY = masks['M_order_family']
    Config.M_FAMILY_GENUS = masks['M_family_genus']
    
    # Convert to numpy for statistics
    M_class_order = masks['M_class_order'].numpy()
    M_order_family = masks['M_order_family'].numpy()
    M_family_genus = masks['M_family_genus'].numpy()
    
    # Display hierarchical constraints
    co_valid = M_class_order.sum()
    co_total = Config.NUM_CLASSES * Config.NUM_ORDERS
    of_valid = M_order_family.sum()
    of_total = Config.NUM_ORDERS * Config.NUM_FAMILIES
    fg_valid = M_family_genus.sum()
    fg_total = Config.NUM_FAMILIES * Config.NUM_GENERA
    
    return df, class_encoder, order_encoder, family_encoder, genus_encoder

def create_data_loaders(df, class_encoder, order_encoder, family_encoder, genus_encoder):
    """Create data loaders"""
    # Use standardized transforms
    train_transforms = TrainingConfig.get_train_transforms()
    val_transforms = TrainingConfig.get_val_test_transforms()
    
    train_df, val_df, test_df = load_split_manifests(Config.DATA_ROOT, Config.MODEL_TYPE)
    validate_images(pd.concat([train_df, val_df]), Config.IMAGES_DIR,
                    Config.OUTPUT_DIR / "preflight" / f"{Config.MODEL_TYPE}_train_validation_images.csv")
    
    train_dataset = HCOFGDataset(train_df, class_encoder, order_encoder, family_encoder, genus_encoder, train_transforms)
    val_dataset = HCOFGDataset(val_df, class_encoder, order_encoder, family_encoder, genus_encoder, val_transforms)
    test_dataset = HCOFGDataset(test_df, class_encoder, order_encoder, family_encoder, genus_encoder, val_transforms)
    
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
    family_loss_total = 0
    genus_loss_total = 0
    class_correct = 0
    order_correct = 0
    family_correct = 0
    genus_correct = 0
    total_samples = 0
    
    for images, class_labels, order_labels, family_labels, genus_labels in train_loader:
        images = images.to(device)
        class_labels = class_labels.to(device)
        order_labels = order_labels.to(device)
        family_labels = family_labels.to(device)
        genus_labels = genus_labels.to(device)
        
        optimizer.zero_grad()
        class_logits, order_logits, family_logits, genus_logits, class_probs, order_probs, family_probs = model(images)
        
        # Class loss (Focal Loss)
        class_loss = criterion(class_logits, class_labels)
        
        # Order loss (Masked Focal Loss)
        m_order = Config.M_CLASS_ORDER.to(device)[class_labels]
        order_loss = masked_focal_loss(order_logits, order_labels, m_order, alpha=TrainingConfig.FOCAL_ALPHA, gamma=TrainingConfig.FOCAL_GAMMA)
        
        # Family loss (Masked Focal Loss)
        m_family = Config.M_ORDER_FAMILY.to(device)[order_labels]
        family_loss = masked_focal_loss(family_logits, family_labels, m_family, alpha=TrainingConfig.FOCAL_ALPHA, gamma=TrainingConfig.FOCAL_GAMMA)
        
        # Genus loss (Masked Focal Loss)
        m_genus = Config.M_FAMILY_GENUS.to(device)[family_labels]
        genus_loss = masked_focal_loss(genus_logits, genus_labels, m_genus, alpha=TrainingConfig.FOCAL_ALPHA, gamma=TrainingConfig.FOCAL_GAMMA)
        
        # Get loss weights from config
        weights = TrainingConfig.LOSS_WEIGHTS["H-COFG"]
        total_loss_batch = (weights["class"] * class_loss + 
                           weights["order"] * order_loss + 
                           weights["family"] * family_loss + 
                           weights["genus"] * genus_loss)
        
        total_loss_batch.backward()
        optimizer.step()
        
        total_loss += total_loss_batch.item()
        class_loss_total += class_loss.item()
        order_loss_total += order_loss.item()
        family_loss_total += family_loss.item()
        genus_loss_total += genus_loss.item()
        
        with torch.no_grad():
            pred_class, pred_order, pred_family, pred_genus = hierarchical_predict(
                class_logits.detach(), order_logits.detach(), 
                family_logits=family_logits.detach(), genus_logits=genus_logits.detach(),
                M_class_order=Config.M_CLASS_ORDER, M_order_family=Config.M_ORDER_FAMILY, 
                M_family_genus=Config.M_FAMILY_GENUS
            )
        
        class_correct += (pred_class == class_labels).sum().item()
        order_correct += (pred_order == order_labels).sum().item()
        family_correct += (pred_family == family_labels).sum().item()
        genus_correct += (pred_genus == genus_labels).sum().item()
        total_samples += images.size(0)
        
    
    num_batches = len(train_loader)
    avg_loss = total_loss / num_batches
    class_acc = class_correct / total_samples
    order_acc = order_correct / total_samples
    family_acc = family_correct / total_samples
    genus_acc = genus_correct / total_samples
    avg_class_loss = class_loss_total / num_batches
    avg_order_loss = order_loss_total / num_batches
    avg_family_loss = family_loss_total / num_batches
    avg_genus_loss = genus_loss_total / num_batches
    
    return (
        avg_loss,
        class_acc,
        order_acc,
        family_acc,
        genus_acc,
        avg_class_loss,
        avg_order_loss,
        avg_family_loss,
        avg_genus_loss,
    )

# hierarchical_predict is now imported from utils.hierarchical_predict

def validate_epoch(model, val_loader, criterion, device):
    """Validate one epoch"""
    model.eval()
    total_loss = 0
    class_loss_total = 0
    order_loss_total = 0
    family_loss_total = 0
    genus_loss_total = 0
    class_correct = 0
    order_correct = 0
    family_correct = 0
    genus_correct = 0
    total_samples = 0
    
    # For F1 calculation
    all_class_preds = []
    all_class_labels = []
    all_order_preds = []
    all_order_labels = []
    all_family_preds = []
    all_family_labels = []
    all_genus_preds = []
    all_genus_labels = []
    
    with torch.no_grad():
        for images, class_labels, order_labels, family_labels, genus_labels in val_loader:
            images = images.to(device)
            class_labels = class_labels.to(device)
            order_labels = order_labels.to(device)
            family_labels = family_labels.to(device)
            genus_labels = genus_labels.to(device)
            
            class_logits, order_logits, family_logits, genus_logits, class_probs, order_probs, family_probs = model(images)
            
            # Losses
            class_loss = criterion(class_logits, class_labels)
            m_order = Config.M_CLASS_ORDER.to(device)[class_labels]
            order_loss = masked_focal_loss(order_logits, order_labels, m_order, alpha=TrainingConfig.FOCAL_ALPHA, gamma=TrainingConfig.FOCAL_GAMMA)
            m_family = Config.M_ORDER_FAMILY.to(device)[order_labels]
            family_loss = masked_focal_loss(family_logits, family_labels, m_family, alpha=TrainingConfig.FOCAL_ALPHA, gamma=TrainingConfig.FOCAL_GAMMA)
            m_genus = Config.M_FAMILY_GENUS.to(device)[family_labels]
            genus_loss = masked_focal_loss(genus_logits, genus_labels, m_genus, alpha=TrainingConfig.FOCAL_ALPHA, gamma=TrainingConfig.FOCAL_GAMMA)
            
            # Get loss weights from config
            weights = TrainingConfig.LOSS_WEIGHTS["H-COFG"]
            total_loss_batch = (weights["class"] * class_loss + 
                               weights["order"] * order_loss + 
                               weights["family"] * family_loss + 
                               weights["genus"] * genus_loss)
            total_loss += total_loss_batch.item()
            class_loss_total += class_loss.item()
            order_loss_total += order_loss.item()
            family_loss_total += family_loss.item()
            genus_loss_total += genus_loss.item()
            
            # Greedy hierarchical prediction (with mask constraints)
            pred_class, pred_order, pred_family, pred_genus = hierarchical_predict(
                class_logits, order_logits, 
                family_logits=family_logits, genus_logits=genus_logits,
                M_class_order=Config.M_CLASS_ORDER, M_order_family=Config.M_ORDER_FAMILY, 
                M_family_genus=Config.M_FAMILY_GENUS
            )
            
            class_correct += (pred_class == class_labels).sum().item()
            order_correct += (pred_order == order_labels).sum().item()
            family_correct += (pred_family == family_labels).sum().item()
            genus_correct += (pred_genus == genus_labels).sum().item()
            total_samples += images.size(0)
            
            # Collect for F1 calculation
            all_class_preds.extend(pred_class.cpu().numpy())
            all_class_labels.extend(class_labels.cpu().numpy())
            all_order_preds.extend(pred_order.cpu().numpy())
            all_order_labels.extend(order_labels.cpu().numpy())
            all_family_preds.extend(pred_family.cpu().numpy())
            all_family_labels.extend(family_labels.cpu().numpy())
            all_genus_preds.extend(pred_genus.cpu().numpy())
            all_genus_labels.extend(genus_labels.cpu().numpy())
    
    num_batches = len(val_loader)
    avg_loss = total_loss / num_batches
    class_acc = class_correct / total_samples
    order_acc = order_correct / total_samples
    family_acc = family_correct / total_samples
    genus_acc = genus_correct / total_samples
    avg_class_loss = class_loss_total / num_batches
    avg_order_loss = order_loss_total / num_batches
    avg_family_loss = family_loss_total / num_batches
    avg_genus_loss = genus_loss_total / num_batches
    
    # Calculate weighted F1 scores
    class_f1 = f1_score(all_class_labels, all_class_preds, average='weighted', zero_division=0)
    order_f1 = f1_score(all_order_labels, all_order_preds, average='weighted', zero_division=0)
    family_f1 = f1_score(all_family_labels, all_family_preds, average='weighted', zero_division=0)
    genus_f1 = f1_score(all_genus_labels, all_genus_preds, average='weighted', zero_division=0)
    
    return (
        avg_loss,
        class_acc,
        order_acc,
        family_acc,
        genus_acc,
        class_f1,
        order_f1,
        family_f1,
        genus_f1,
        avg_class_loss,
        avg_order_loss,
        avg_family_loss,
        avg_genus_loss,
    )

def main():
    parser = argparse.ArgumentParser(description='Train H-COFG hierarchical model')
    
    # Print separator for model identification
    print("\n" + "=" * 80)
    print(f"Training Model: {Config.MODEL_TYPE}")
    print("=" * 80 + "\n")
    TrainingConfig.set_global_seed()
    
    # Load and prepare data
    df, class_encoder, order_encoder, family_encoder, genus_encoder = load_and_prepare_data()
    
    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(
        df, class_encoder, order_encoder, family_encoder, genus_encoder
    )
    
    # Print training configuration
    print_training_config(Config.MODEL_TYPE, Config, TrainingConfig)
    
    # Create model
    model = HCOFGModel(
        Config.NUM_CLASSES,
        Config.NUM_ORDERS,
        Config.NUM_FAMILIES,
        Config.NUM_GENERA,
        Config.MODEL_NAME,
        pretrained=False,
    )
    load_backbone_from_checkpoint(
        Config.PRETRAINED_MODEL_PATH,
        model.backbone,
    )
    model = model.to(Config.DEVICE)
    
    # Optimizer and scheduler
    # All models use Focal Loss for fair comparison
    criterion = FocalLoss(alpha=TrainingConfig.FOCAL_ALPHA, gamma=TrainingConfig.FOCAL_GAMMA)
    optimizer = optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY)
    scheduler = TrainingConfig.get_lr_scheduler(optimizer)  # ReduceLROnPlateau
    
    # Training history
    history = {
        'train_loss': [], 'val_loss': [],
        'train_class_loss': [], 'val_class_loss': [],
        'train_order_loss': [], 'val_order_loss': [],
        'train_family_loss': [], 'val_family_loss': [],
        'train_genus_loss': [], 'val_genus_loss': [],
        'train_class_acc': [], 'val_class_acc': [],
        'train_order_acc': [], 'val_order_acc': [],
        'train_family_acc': [], 'val_family_acc': [],
        'train_genus_acc': [], 'val_genus_acc': [],
        'val_class_f1': [], 'val_order_f1': [], 'val_family_f1': [], 'val_genus_f1': []
    }
    
    best_val_f1 = 0
    best_epoch = 0
    patience_counter = 0
    
    pbar = tqdm(range(Config.EPOCHS), desc="Training")
    
    for epoch in pbar:
        # Train
        (
            train_loss,
            train_class_acc,
            train_order_acc,
            train_family_acc,
            train_genus_acc,
            train_class_loss,
            train_order_loss,
            train_family_loss,
            train_genus_loss,
        ) = train_epoch(
            model, train_loader, criterion, optimizer, Config.DEVICE
        )
        
        # Validate
        (
            val_loss,
            val_class_acc,
            val_order_acc,
            val_family_acc,
            val_genus_acc,
            val_class_f1,
            val_order_f1,
            val_family_f1,
            val_genus_f1,
            val_class_loss,
            val_order_loss,
            val_family_loss,
            val_genus_loss,
        ) = validate_epoch(
            model, val_loader, criterion, Config.DEVICE
        )
        
        # Update history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_class_loss'].append(train_class_loss)
        history['val_class_loss'].append(val_class_loss)
        history['train_order_loss'].append(train_order_loss)
        history['val_order_loss'].append(val_order_loss)
        history['train_family_loss'].append(train_family_loss)
        history['val_family_loss'].append(val_family_loss)
        history['train_genus_loss'].append(train_genus_loss)
        history['val_genus_loss'].append(val_genus_loss)
        history['train_class_acc'].append(train_class_acc)
        history['val_class_acc'].append(val_class_acc)
        history['train_order_acc'].append(train_order_acc)
        history['val_order_acc'].append(val_order_acc)
        history['train_family_acc'].append(train_family_acc)
        history['val_family_acc'].append(val_family_acc)
        history['train_genus_acc'].append(train_genus_acc)
        history['val_genus_acc'].append(val_genus_acc)
        history['val_class_f1'].append(val_class_f1)
        history['val_order_f1'].append(val_order_f1)
        history['val_family_f1'].append(val_family_f1)
        history['val_genus_f1'].append(val_genus_f1)
        
        # Update learning rate (ReduceLROnPlateau needs val_loss)
        scheduler.step(val_loss)
        
        # Check if best model
        is_best = val_genus_f1 > best_val_f1
        if is_best:
            best_val_f1 = val_genus_f1
            best_epoch = epoch + 1
            patience_counter = 0
            
            # Save best model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_genus_f1': val_genus_f1,
                'val_genus_acc': val_genus_acc,
                'val_family_f1': val_family_f1,
                'val_order_f1': val_order_f1,
                'val_class_f1': val_class_f1,
                'class_names': list(class_encoder.classes_),
                'order_names': list(order_encoder.classes_),
                'family_names': list(family_encoder.classes_),
                'genus_names': list(genus_encoder.classes_),
                'num_classes': Config.NUM_CLASSES,
                'num_orders': Config.NUM_ORDERS,
                'num_families': Config.NUM_FAMILIES,
                'num_genera': Config.NUM_GENERA,
                'M_class_order': Config.M_CLASS_ORDER,
                'M_order_family': Config.M_ORDER_FAMILY,
                'M_family_genus': Config.M_FAMILY_GENUS,
                'checkpoint_schema_version': CHECKPOINT_SCHEMA_VERSION
            }, Config.CHECKPOINT_DIR / 'best_H_COFG_model.pth')
            
            pbar.set_postfix({'best_f1': f'{val_genus_f1:.4f}', 'epoch': epoch+1})
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= Config.PATIENCE:
            break
    
    print(f"\n✅ Training complete: Best Genus F1={best_val_f1:.4f} at epoch {best_epoch}/{len(history['train_loss'])}")
    
    # Save training history
    history['best_val_genus_f1'] = best_val_f1
    history['best_epoch'] = best_epoch
    
    history_path = Config.LOG_DIR / 'H_COFG__training_history.json'
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)

if __name__ == "__main__":
    main()
