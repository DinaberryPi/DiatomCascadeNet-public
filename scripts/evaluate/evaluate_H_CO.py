#!/usr/bin/env python3
"""
DiatomScanNet Level 2 Evaluation: Class → Order
Evaluate the two-level hierarchical model on test set
"""

import os
import sys
import json
import argparse
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import timm
from tqdm import tqdm

# Add project root to path for utils
from diatom_cascade.prediction import (
    greedy_hierarchical_predict as hierarchical_predict,
    beam_search_hierarchical_predict,
    level_wise_argmax_predict as argmax_independent_predict
)
from diatom_cascade.data.integrity import load_split_manifests, validate_images
from diatom_cascade.runtime import load_checkpoint
from diatom_cascade.evaluation import create_test_loader
from diatom_cascade.runtime import get_preprocess_transforms

# Use unified evaluation configuration
from diatom_cascade.config.evaluation_config import EvaluationConfig as Config
from diatom_cascade.models import ClassToOrderModel

# Override checkpoint path for this model
Config.CHECKPOINT_PATH = Config.get_checkpoint_path("H-CO")

class ClassToOrderDataset(Dataset):
    """Dataset for Class → Order hierarchical classification"""
    def __init__(self, df, class_encoder, order_encoder, transforms=None):
        self.df = df.reset_index(drop=True)
        self.transforms = transforms
        self.class_encoder = class_encoder
        self.order_encoder = order_encoder
        
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
        
        # Encode labels
        class_label = int(self.class_encoder.transform([row['class']])[0])
        order_label = int(self.order_encoder.transform([row['order']])[0])
        
        return image, class_label, order_label


# topdown_predict is now replaced by hierarchical_predict from utils.hierarchical_predict

def evaluate_model(model, test_loader, M_class_order, device):
    """Evaluate model on test set"""
    model.eval()
    
    # Greedy hierarchical prediction results (with mask constraints)
    all_class_labels = []
    all_order_labels = []
    all_class_pred_hier = []
    all_order_pred_hier = []
    
    # Level-wise argmax prediction results (no mask constraints, for comparison only)
    all_class_pred_argmax = []
    all_order_pred_argmax = []
    
    # Beam search prediction results (globally constrained)
    all_class_pred_beam = []
    all_order_pred_beam = []
    
    with torch.no_grad():
        for images, class_labels, order_labels in tqdm(test_loader, desc="Testing"):
            images = images.to(device)
            class_logits, order_logits, _ = model(images)
            
            # Method 1: Greedy hierarchical prediction (with mask constraints)
            pred_class_hier, pred_order_hier = hierarchical_predict(
                class_logits, order_logits, 
                M_class_order=M_class_order
            )
            
            # Method 2: Level-wise argmax (no mask constraints, for comparison only)
            pred_class_argmax, pred_order_argmax = argmax_independent_predict(
                class_logits, order_logits
            )
            
            # Method 3: Beam search prediction (globally constrained)
            pred_class_beam, pred_order_beam = beam_search_hierarchical_predict(
                class_logits, order_logits,
                M_class_order=M_class_order,
                beam_width=Config.BEAM_WIDTH
            )
            
            # Collect true labels
            all_class_labels.extend(class_labels.cpu().numpy())
            all_order_labels.extend(order_labels.cpu().numpy())
            
            # Collect greedy hierarchical predictions (with mask constraints)
            all_class_pred_hier.extend(pred_class_hier.cpu().numpy())
            all_order_pred_hier.extend(pred_order_hier.cpu().numpy())
            
            # Collect argmax predictions
            all_class_pred_argmax.extend(pred_class_argmax.cpu().numpy())
            all_order_pred_argmax.extend(pred_order_argmax.cpu().numpy())
            
            # Collect beam search predictions
            all_class_pred_beam.extend(pred_class_beam.cpu().numpy())
            all_order_pred_beam.extend(pred_order_beam.cpu().numpy())
    
    # For backward compatibility, use greedy hierarchical predictions (with mask constraints) as default
    all_class_preds = all_class_pred_hier
    all_order_preds = all_order_pred_hier
    
    # Calculate metrics for greedy hierarchical prediction (with mask constraints)
    class_acc = accuracy_score(all_class_labels, all_class_preds)
    class_f1_macro = f1_score(all_class_labels, all_class_preds, average='macro')
    class_f1_weighted = f1_score(all_class_labels, all_class_preds, average='weighted')
    
    order_acc = accuracy_score(all_order_labels, all_order_preds)
    order_f1_macro = f1_score(all_order_labels, all_order_preds, average='macro')
    order_f1_weighted = f1_score(all_order_labels, all_order_preds, average='weighted')
    
    # Calculate metrics for level-wise argmax prediction (no mask constraints)
    class_acc_argmax = accuracy_score(all_class_labels, all_class_pred_argmax)
    class_f1_argmax = f1_score(all_class_labels, all_class_pred_argmax, average='weighted', zero_division=0)
    order_acc_argmax = accuracy_score(all_order_labels, all_order_pred_argmax)
    order_f1_argmax = f1_score(all_order_labels, all_order_pred_argmax, average='weighted', zero_division=0)
    
    # Calculate metrics for beam search prediction
    class_acc_beam = accuracy_score(all_class_labels, all_class_pred_beam)
    class_f1_beam = f1_score(all_class_labels, all_class_pred_beam, average='weighted', zero_division=0)
    order_acc_beam = accuracy_score(all_order_labels, all_order_pred_beam)
    order_f1_beam = f1_score(all_order_labels, all_order_pred_beam, average='weighted', zero_division=0)
    
    # Calculate agreement rates
    class_agreement_hier_argmax = np.mean(np.array(all_class_pred_hier) == np.array(all_class_pred_argmax))
    order_agreement_hier_argmax = np.mean(np.array(all_order_pred_hier) == np.array(all_order_pred_argmax))
    
    class_agreement_hier_beam = np.mean(np.array(all_class_pred_hier) == np.array(all_class_pred_beam))
    order_agreement_hier_beam = np.mean(np.array(all_order_pred_hier) == np.array(all_order_pred_beam))
    
    num_test_samples = len(all_class_labels)
    print(f"Greedy: C {class_acc:.4f} ({class_f1_weighted:.4f}) | O {order_acc:.4f} ({order_f1_weighted:.4f}) (N={num_test_samples})")
    print(f"Argmax: C {class_acc_argmax:.4f} ({class_f1_argmax:.4f}) | O {order_acc_argmax:.4f} ({order_f1_argmax:.4f}) (N={num_test_samples})")
    print(f"Beam:   C {class_acc_beam:.4f} ({class_f1_beam:.4f}) | O {order_acc_beam:.4f} ({order_f1_beam:.4f}) (N={num_test_samples})")
    
    return {
        'class': {
            'accuracy': class_acc,
            'f1_macro': class_f1_macro,
            'f1_weighted': class_f1_weighted,
            'predictions': all_class_preds,
            'labels': all_class_labels
        },
        'order': {
            'accuracy': order_acc,
            'f1_macro': order_f1_macro,
            'f1_weighted': order_f1_weighted,
            'predictions': all_order_preds,
            'labels': all_order_labels
        }
    }

# Plotting functions moved to analyze/generate_evaluation_figures.py

def main():
    parser = argparse.ArgumentParser(description='Evaluate H-CO hierarchical model')
    
    # Print separator for model identification
    print("\n" + "=" * 80)
    print("Evaluating Model: H-CO")
    print("=" * 80 + "\n")
    
    # Check if checkpoint exists
    if not Config.CHECKPOINT_PATH.exists():
        print(f"Error: Checkpoint not found at {Config.CHECKPOINT_PATH}")
        print("Please train the model first using: python train/train_H_CO.py")
        return
    
    # Load checkpoint
    # Use unified checkpoint loading
    checkpoint = load_checkpoint(Config.CHECKPOINT_PATH, Config.DEVICE)
    
    # Load data - use the same pre-filtered dataset as training
    # Load model-specific dataset from mapping (same as training)
    import json
    mapping_file = Config.DATA_ROOT / "preprocessed" / "model_data_mapping.json"
    if not mapping_file.exists():
        raise FileNotFoundError(
            f"Model mapping file not found: {mapping_file}\n"
            "Please run: python -m scripts.data.preprocessing.create_filtered_datasets"
        )
    
    with open(mapping_file, 'r', encoding='utf-8') as f:
        model_data_mapping = json.load(f)
    
    model_type = "H-CO"
    if model_type not in model_data_mapping:
        raise ValueError(f"Model type '{model_type}' not found in mapping")
    
    # Load the correct pre-filtered dataset file
    # Handle relative paths (e.g., "../cleaned/labels_clean.csv" for F-C)
    dataset_path = model_data_mapping[model_type]
    dataset_file = (Config.DATA_ROOT / "preprocessed" / dataset_path).resolve()
    Config.LABELS_CSV = dataset_file
    
    print(f"Model: {model_type}")
    print(f"Dataset: {dataset_file.name}")
    
    df = pd.read_csv(Config.LABELS_CSV)
    
    # Get encoders - try to get from checkpoint, or recreate from names
    if 'class_encoder' in checkpoint and 'order_encoder' in checkpoint:
        # Old format: encoder objects saved directly
        class_encoder = checkpoint['class_encoder']
        order_encoder = checkpoint['order_encoder']
    elif 'class_names' in checkpoint and 'order_names' in checkpoint:
        # New format: recreate from names
        class_names = checkpoint['class_names']
        order_names = checkpoint['order_names']
        class_encoder = LabelEncoder()
        class_encoder.classes_ = np.array(class_names)
        order_encoder = LabelEncoder()
        order_encoder.classes_ = np.array(order_names)
    else:
        raise ValueError("Checkpoint missing class_encoder/order_encoder or class_names/order_names. "
                        "Please retrain the model.")
    
    M_class_order = checkpoint['M_class_order']
    
    _, _, test_df = load_split_manifests(Config.DATA_ROOT, "H-CO")
    validate_images(test_df, Config.IMAGES_DIR,
                    Config.OUTPUT_DIR / "preflight" / "H-CO_test_images.csv")
    
    # Data transforms (use unified config)
    test_transforms = get_preprocess_transforms()
    
    # Create test dataset
    test_dataset = ClassToOrderDataset(test_df, class_encoder, order_encoder, test_transforms)
    test_loader = create_test_loader(test_dataset)
    
    # Create model
    num_classes = checkpoint['num_classes']
    num_orders = checkpoint['num_orders']
    model = ClassToOrderModel(
        num_classes,
        num_orders,
        Config.MODEL_NAME,
        pretrained=False,
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(Config.DEVICE)
    model.eval()
    print(f"✅ Model loaded ({num_classes} classes, {num_orders} orders)")
    print(f"Test set: {len(test_df)} samples")
    
    # Evaluate
    results = evaluate_model(model, test_loader, M_class_order, Config.DEVICE)
    
    # Save results
    results_to_save = {
        'class': {
            'accuracy': float(results['class']['accuracy']),
            'f1_macro': float(results['class']['f1_macro']),
            'f1_weighted': float(results['class']['f1_weighted'])
        },
        'order': {
            'accuracy': float(results['order']['accuracy']),
            'f1_macro': float(results['order']['f1_macro']),
            'f1_weighted': float(results['order']['f1_weighted'])
        },
        'test_samples': len(test_df)
    }
    
    output_file = Config.EVAL_DIR / 'H_CO_evaluation_report.json'
    with open(output_file, 'w') as f:
        json.dump(results_to_save, f, indent=2)

if __name__ == "__main__":
    main()
