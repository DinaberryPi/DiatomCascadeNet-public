#!/usr/bin/env python3
"""
DiatomScanNet 纲级别分类测试集评估脚本
- 在测试集上进行最终评估
- 生成详细报告和混淆矩阵
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

# Add project root to path for imports

# Use unified evaluation configuration
from diatom_cascade.config.evaluation_config import EvaluationConfig as Config
from diatom_cascade.evaluation import create_test_loader
from diatom_cascade.data.integrity import load_split_manifests, validate_images
from diatom_cascade.runtime import get_preprocess_transforms, load_checkpoint, load_label_encoder
from diatom_cascade.models import FlatClassifier as EfficientNetClassifier

# Override checkpoint path for this model
Config.CHECKPOINT_PATH = Config.get_checkpoint_path("F-C")

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
        label = str(row['class'])
        label_id = self.label_encoder.transform([label])[0]
        
        return image, label_id, row['class']  # Return original class name


def load_model(checkpoint_path):
    """Load trained model"""
    checkpoint = load_checkpoint(checkpoint_path, Config.DEVICE)
    
    config = checkpoint['config']
    label_encoder = load_label_encoder(checkpoint, 'class_names')
    
    model = EfficientNetClassifier(
        num_classes=config['NUM_CLASSES'],
        model_name=config['MODEL_NAME'],
        pretrained=False,
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(Config.DEVICE)
    model.eval()
    
    print(f"✅ Model loaded ({config['NUM_CLASSES']} classes)")
    
    return model, label_encoder, config

def prepare_test_data(label_encoder):
    """Prepare test data"""
    # Load dataset mapping
    mapping_file = Config.DATA_ROOT / "preprocessed" / "model_data_mapping.json"
    if not mapping_file.exists():
        raise FileNotFoundError(f"Model mapping file not found: {mapping_file}")
    
    with open(mapping_file, 'r', encoding='utf-8') as f:
        model_data_mapping = json.load(f)
    
    model_type = "F-C"
    if model_type not in model_data_mapping:
        available_keys = list(model_data_mapping.keys())
        raise ValueError(
            f"Model type '{model_type}' not found in mapping. "
            f"Available keys: {available_keys}. "
            f"Mapping file: {mapping_file}"
        )
    
    # Load the correct pre-filtered dataset file
    # Handle relative paths (e.g., "../cleaned/labels_clean.csv" for F-C)
    dataset_path = model_data_mapping[model_type]
    dataset_file = (Config.DATA_ROOT / "preprocessed" / dataset_path).resolve()
    Config.LABELS_CSV = dataset_file
    
    print(f"Model: {model_type}")
    print(f"Dataset: {dataset_file.name}")
    
    # Load all data
    df = pd.read_csv(Config.LABELS_CSV)
    df = df[df['class'].notna() & (df['class'] != '')].copy()
    
    # Merge Mediophyceae into Coscinodiscophyceae
    df.loc[df['class'] == 'Mediophyceae', 'class'] = 'Coscinodiscophyceae'
    
    _, _, test_df = load_split_manifests(Config.DATA_ROOT, model_type)
    validate_images(test_df, Config.IMAGES_DIR,
                    Config.OUTPUT_DIR / "preflight" / f"{model_type}_test_images.csv")
    
    print(f"Test set: {len(test_df)} samples")
    
    # Data transforms (use unified config)
    test_transforms = get_preprocess_transforms()
    
    # Create dataset
    test_dataset = DiatomDataset(test_df, label_encoder, test_transforms)
    
    # Create data loader (use unified config)
    test_loader = create_test_loader(test_dataset)
    
    return test_loader, test_df

def evaluate_model(model, test_loader, label_encoder):
    """Evaluate model"""
    model.eval()
    all_preds = []
    all_labels = []
    all_true_classes = []
    all_pred_classes = []
    all_probabilities = []
    
    with torch.no_grad():
        for images, labels, true_classes in tqdm(test_loader, desc="Evaluating"):
            images = images.to(Config.DEVICE)
            labels = labels.to(Config.DEVICE)
            
            outputs = model(images)
            probabilities = torch.softmax(outputs, dim=1)
            preds = torch.argmax(outputs, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_true_classes.extend(true_classes)
            all_pred_classes.extend([label_encoder.classes_[p] for p in preds.cpu().numpy()])
            all_probabilities.extend(probabilities.cpu().numpy())
    
    return all_preds, all_labels, all_true_classes, all_pred_classes, all_probabilities

def generate_report(true_classes, pred_classes, label_encoder, save_path):
    """Generate evaluation report"""
    # Calculate metrics
    accuracy = accuracy_score(true_classes, pred_classes)
    f1_macro = f1_score(true_classes, pred_classes, average='macro')
    f1_weighted = f1_score(true_classes, pred_classes, average='weighted')
    
    # Classification report
    report = classification_report(
        true_classes, pred_classes, 
        target_names=label_encoder.classes_,
        output_dict=True
    )
    
    # Always print main results (concise)
    num_test_samples = len(true_classes)
    print(f"Results: Acc {accuracy:.4f} ({accuracy*100:.2f}%), F1-macro {f1_macro:.4f}, F1-weighted {f1_weighted:.4f} (N={num_test_samples})")
    
    # Save report
    report_data = {
        'overall_metrics': {
            'accuracy': accuracy,
            'f1_macro': f1_macro,
            'f1_weighted': f1_weighted
        },
        'per_class_metrics': report,
        'test_samples': len(true_classes)  # Add test set size
    }
    
    with open(save_path / 'F_C_evaluation_report.json', 'w') as f:
        json.dump(report_data, f, indent=2)
    
    return report

# Plotting functions moved to analyze/generate_evaluation_figures.py

def main():
    parser = argparse.ArgumentParser(description='Evaluate F-C flat model')
    
    # Print separator for model identification
    print("\n" + "=" * 80)
    print("Evaluating Model: F-C")
    print("=" * 80 + "\n")
    
    # Check model file
    if not Path(Config.CHECKPOINT_PATH).exists():
        print(f"Model file not found: {Config.CHECKPOINT_PATH}")
        print("Please train the model first: python -m scripts.train.train_F_C")
        return
    
    # Load model
    model, label_encoder, config = load_model(Config.CHECKPOINT_PATH)
    
    # Prepare test data
    test_loader, test_df = prepare_test_data(label_encoder)
    
    # Evaluate model
    all_preds, all_labels, all_true_classes, all_pred_classes, all_probabilities = evaluate_model(
        model, test_loader, label_encoder
    )
    
    # Generate report
    report = generate_report(
        all_true_classes, all_pred_classes, label_encoder, Config.EVAL_DIR
    )
    
    report_path = Config.EVAL_DIR / 'F_C_evaluation_report.json'

if __name__ == "__main__":
    main()
