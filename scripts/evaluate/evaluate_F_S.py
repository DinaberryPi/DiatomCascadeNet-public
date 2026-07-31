#!/usr/bin/env python3
"""
DiatomScanNet F Species Baseline Prediction
- Direct classification of Species without hierarchical structure
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
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, classification_report
from tqdm import tqdm
import timm

# Add project root to path for imports

# Use unified evaluation configuration
from diatom_cascade.config.evaluation_config import EvaluationConfig as Config
from diatom_cascade.evaluation import create_test_loader
from diatom_cascade.data.integrity import load_split_manifests, validate_images
from diatom_cascade.runtime import get_preprocess_transforms, load_checkpoint
from diatom_cascade.prediction import load_taxonomy_tree, flat_bottom_up_lookup_from_species as infer_upper_levels_from_species
from diatom_cascade.models import FlatClassifier

# Override checkpoint path for this model
Config.CHECKPOINT_PATH = Config.get_checkpoint_path("F-S")


def load_model(checkpoint_path):
    """Load trained F Species model (uses unified checkpoint loading)"""
    # Use unified checkpoint loading
    checkpoint = load_checkpoint(checkpoint_path, Config.DEVICE)
    
    # Check checkpoint structure - F-S checkpoint may have different formats
    if 'config' in checkpoint and 'label_encoder' in checkpoint:
        # New format: has config and label_encoder
        config = checkpoint['config']
        label_encoder = checkpoint['label_encoder']
    elif 'label_encoder' in checkpoint:
        # Has label_encoder but no config
        label_encoder = checkpoint['label_encoder']
        if hasattr(label_encoder, 'classes_') and len(label_encoder.classes_) > 0:
            num_classes = len(label_encoder.classes_)
        else:
            num_classes = checkpoint.get('num_classes') or checkpoint.get('num_species')
            if num_classes is None:
                raise ValueError("Cannot determine number of classes from checkpoint")
        config = {
            'NUM_CLASSES': num_classes,
            'MODEL_NAME': Config.MODEL_NAME
        }
    elif 'species_names' in checkpoint:
        # F-S format: has species_names list, need to rebuild label_encoder
        species_names = checkpoint['species_names']
        num_classes = checkpoint.get('num_species') or len(species_names)
        
        # Rebuild label_encoder from species_names
        label_encoder = LabelEncoder()
        label_encoder.fit(species_names)  # This sets label_encoder.classes_
        
        config = {
            'NUM_CLASSES': num_classes,
            'MODEL_NAME': Config.MODEL_NAME
        }
    else:
        raise ValueError("Cannot determine checkpoint format. Missing required keys.")
    
    model = FlatClassifier(
        num_classes=config['NUM_CLASSES'],
        model_name=config.get('MODEL_NAME', Config.MODEL_NAME),
        pretrained=False,
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(Config.DEVICE)
    model.eval()
    
    # Verify model config matches evaluation config
    if config.get('MODEL_NAME') != Config.MODEL_NAME:
        print(f"⚠️  Warning: Model was trained with {config.get('MODEL_NAME')}, but evaluation uses {Config.MODEL_NAME}")
    if config.get('IMAGE_SIZE') and config.get('IMAGE_SIZE') != Config.IMAGE_SIZE:
        print(f"⚠️  Warning: Model was trained with IMAGE_SIZE={config.get('IMAGE_SIZE')}, but evaluation uses {Config.IMAGE_SIZE}")
    
    print(f"✅ Model loaded ({config['NUM_CLASSES']} species)")
    
    return model, label_encoder, config

def preprocess_image(image_path, image_size=None):
    """Preprocess image (uses unified config)"""
    if image_size is None:
        image_size = Config.IMAGE_SIZE
    image = Image.open(image_path).convert('RGB')
    transform = get_preprocess_transforms()  # Use unified transforms
    image_tensor = transform(image).unsqueeze(0)
    return image_tensor, image

class DiatomDataset(Dataset):
    """Dataset for species-level classification"""
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
        species = str(row['species'])
        label_id = self.label_encoder.transform([species])[0]
        
        # Return image, label_id, original species name, and upper levels if available
        result = [image, label_id, species]
        if 'class' in row and pd.notna(row['class']):
            result.append(str(row['class']))
        else:
            result.append('')
        if 'order' in row and pd.notna(row['order']):
            result.append(str(row['order']))
        else:
            result.append('')
        if 'family' in row and pd.notna(row['family']):
            result.append(str(row['family']))
        else:
            result.append('')
        if 'genus' in row and pd.notna(row['genus']):
            result.append(str(row['genus']))
        else:
            result.append('')
        
        return tuple(result)

def prepare_test_data(label_encoder):
    """Prepare test data for species-level evaluation"""
    # Load dataset mapping
    mapping_file = Config.DATA_ROOT / "preprocessed" / "model_data_mapping.json"
    with open(mapping_file, 'r', encoding='utf-8') as f:
        model_data_mapping = json.load(f)
    
    model_type = "F-S"
    if model_type not in model_data_mapping:
        raise ValueError(f"Model type '{model_type}' not found in mapping")
    
    # Load the correct pre-filtered dataset file
    # Handle relative paths (e.g., "../cleaned/labels_clean.csv" for F-C)
    dataset_path = model_data_mapping[model_type]
    dataset_file = (Config.DATA_ROOT / "preprocessed" / dataset_path).resolve()
    Config.LABELS_CSV = dataset_file
    
    print(f"Model: {model_type}")
    print(f"Dataset: {dataset_file.name}")
    
    # Load all data
    df = pd.read_csv(Config.LABELS_CSV)
    df = df[df['species'].notna() & (df['species'] != '')].copy()
    
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

# Taxonomy lookup functions are imported from utils.predict

def evaluate_model(model, test_loader, label_encoder):
    """Evaluate F Species model on test set, including upper level lookup from taxonomy tree"""
    model.eval()
    all_preds = []
    all_labels = []
    all_true_labels = []
    all_pred_labels = []
    all_probabilities = []
    all_pred_classes = []
    all_pred_orders = []
    all_pred_families = []
    all_pred_genera = []
    all_true_classes = []
    all_true_orders = []
    all_true_families = []
    all_true_genera = []
    
    # Load taxonomy tree if available (use unified function)
    taxonomy_tree, mappings = load_taxonomy_tree(Config.TAXONOMY_JSON)
    has_taxonomy = taxonomy_tree is not None
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            images = batch[0].to(Config.DEVICE)
            labels = batch[1].to(Config.DEVICE)
            true_species_labels = batch[2]  # Original species names
            
            # Get true upper levels (if available)
            true_classes_batch = batch[3] if len(batch) > 3 else []
            true_orders_batch = batch[4] if len(batch) > 4 else []
            true_families_batch = batch[5] if len(batch) > 5 else []
            true_genera_batch = batch[6] if len(batch) > 6 else []
            
            outputs = model(images)
            probabilities = torch.softmax(outputs, dim=1)
            preds = torch.argmax(outputs, dim=1)
            
            batch_size = len(preds)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_true_labels.extend(true_species_labels)
            all_pred_labels.extend([label_encoder.classes_[p] for p in preds.cpu().numpy()])
            all_probabilities.extend(probabilities.cpu().numpy())
            
            # Look up upper levels from predicted species using taxonomy tree
            if has_taxonomy:
                for i in range(batch_size):
                    pred_species = label_encoder.classes_[preds[i].item()]
                    pred_class, pred_order, pred_family, pred_genus = infer_upper_levels_from_species(
                        pred_species, taxonomy_tree, mappings
                    )
                    
                    all_pred_classes.append(pred_class)
                    all_pred_orders.append(pred_order)
                    all_pred_families.append(pred_family)
                    all_pred_genera.append(pred_genus)
                    
                    # Get true upper levels
                    if true_classes_batch and len(true_classes_batch) > i:
                        true_class = true_classes_batch[i] if isinstance(true_classes_batch, (list, tuple)) else ''
                        all_true_classes.append(true_class if true_class else '')
                    else:
                        all_true_classes.append('')
                    
                    if true_orders_batch and len(true_orders_batch) > i:
                        true_order = true_orders_batch[i] if isinstance(true_orders_batch, (list, tuple)) else ''
                        all_true_orders.append(true_order if true_order else '')
                    else:
                        all_true_orders.append('')
                    
                    if true_families_batch and len(true_families_batch) > i:
                        true_family = true_families_batch[i] if isinstance(true_families_batch, (list, tuple)) else ''
                        all_true_families.append(true_family if true_family else '')
                    else:
                        all_true_families.append('')
                    
                    if true_genera_batch and len(true_genera_batch) > i:
                        true_genus = true_genera_batch[i] if isinstance(true_genera_batch, (list, tuple)) else ''
                        all_true_genera.append(true_genus if true_genus else '')
                    else:
                        all_true_genera.append('')
            else:
                # No taxonomy tree, fill with empty values
                for i in range(batch_size):
                    all_pred_classes.append(None)
                    all_pred_orders.append(None)
                    all_pred_families.append(None)
                    all_pred_genera.append(None)
                    all_true_classes.append('')
                    all_true_orders.append('')
                    all_true_families.append('')
                    all_true_genera.append('')
    
    return {
        'all_preds': all_preds,
        'all_labels': all_labels,
        'all_true_labels': all_true_labels,
        'all_pred_labels': all_pred_labels,
        'all_probabilities': all_probabilities,
        'all_pred_classes': all_pred_classes,
        'all_pred_orders': all_pred_orders,
        'all_pred_families': all_pred_families,
        'all_pred_genera': all_pred_genera,
        'all_true_classes': all_true_classes,
        'all_true_orders': all_true_orders,
        'all_true_families': all_true_families,
        'all_true_genera': all_true_genera,
        'has_taxonomy': has_taxonomy
    }

def generate_report(eval_results, label_encoder, save_path):
    """Generate evaluation report from evaluation results"""
    # Extract results
    true_species_labels = eval_results['all_true_labels']
    pred_species_labels = eval_results['all_pred_labels']
    
    # Calculate species-level metrics
    species_accuracy = accuracy_score(eval_results['all_labels'], eval_results['all_preds'])
    species_f1_macro = f1_score(eval_results['all_labels'], eval_results['all_preds'], average='macro', zero_division=0)
    species_f1_weighted = f1_score(eval_results['all_labels'], eval_results['all_preds'], average='weighted', zero_division=0)
    
    # Always print main results (concise)
    num_test_samples = len(true_species_labels)
    print(f"Species: Acc {species_accuracy:.4f} ({species_accuracy*100:.2f}%), F1-macro {species_f1_macro:.4f}, F1-weighted {species_f1_weighted:.4f} (N={num_test_samples})")
    
    # Upper level metrics (if taxonomy tree is available)
    upper_level_metrics = {}
    if eval_results['has_taxonomy']:
        # Filter out None/empty predictions and labels
        true_classes = [t for t in eval_results['all_true_classes'] if t and t != '']
        pred_classes = [p for p, t in zip(eval_results['all_pred_classes'], eval_results['all_true_classes']) if t and t != '' and p]
        
        true_orders = [t for t in eval_results['all_true_orders'] if t and t != '']
        pred_orders = [p for p, t in zip(eval_results['all_pred_orders'], eval_results['all_true_orders']) if t and t != '' and p]
        
        true_families = [t for t in eval_results['all_true_families'] if t and t != '']
        pred_families = [p for p, t in zip(eval_results['all_pred_families'], eval_results['all_true_families']) if t and t != '' and p]
        
        true_genera = [t for t in eval_results['all_true_genera'] if t and t != '']
        pred_genera = [p for p, t in zip(eval_results['all_pred_genera'], eval_results['all_true_genera']) if t and t != '' and p]
        
        if len(true_classes) > 0 and len(pred_classes) > 0:
            class_accuracy = accuracy_score(true_classes, pred_classes)
            class_f1_macro = f1_score(true_classes, pred_classes, average='macro', zero_division=0)
            class_f1_weighted = f1_score(true_classes, pred_classes, average='weighted', zero_division=0)
            upper_level_metrics['class'] = {
                'accuracy': float(class_accuracy),
                'f1_macro': float(class_f1_macro),
                'f1_weighted': float(class_f1_weighted),
                'num_samples': len(true_classes)
            }
            print(f"Class (lookup): Acc {class_accuracy:.4f} ({class_accuracy*100:.2f}%), F1-macro {class_f1_macro:.4f}, F1-weighted {class_f1_weighted:.4f} (N={len(true_classes)})")
        if len(true_orders) > 0 and len(pred_orders) > 0:
            order_accuracy = accuracy_score(true_orders, pred_orders)
            order_f1_macro = f1_score(true_orders, pred_orders, average='macro', zero_division=0)
            order_f1_weighted = f1_score(true_orders, pred_orders, average='weighted', zero_division=0)
            upper_level_metrics['order'] = {
                'accuracy': float(order_accuracy),
                'f1_macro': float(order_f1_macro),
                'f1_weighted': float(order_f1_weighted),
                'num_samples': len(true_orders)
            }
            print(f"Order (lookup): Acc {order_accuracy:.4f} ({order_accuracy*100:.2f}%), F1-macro {order_f1_macro:.4f}, F1-weighted {order_f1_weighted:.4f} (N={len(true_orders)})")
        
        if len(true_families) > 0 and len(pred_families) > 0:
            family_accuracy = accuracy_score(true_families, pred_families)
            family_f1_macro = f1_score(true_families, pred_families, average='macro', zero_division=0)
            family_f1_weighted = f1_score(true_families, pred_families, average='weighted', zero_division=0)
            upper_level_metrics['family'] = {
                'accuracy': float(family_accuracy),
                'f1_macro': float(family_f1_macro),
                'f1_weighted': float(family_f1_weighted),
                'num_samples': len(true_families)
            }
            print(f"Family (lookup): Acc {family_accuracy:.4f} ({family_accuracy*100:.2f}%), F1-macro {family_f1_macro:.4f}, F1-weighted {family_f1_weighted:.4f} (N={len(true_families)})")
        
        if len(true_genera) > 0 and len(pred_genera) > 0:
            genus_accuracy = accuracy_score(true_genera, pred_genera)
            genus_f1_macro = f1_score(true_genera, pred_genera, average='macro', zero_division=0)
            genus_f1_weighted = f1_score(true_genera, pred_genera, average='weighted', zero_division=0)
            upper_level_metrics['genus'] = {
                'accuracy': float(genus_accuracy),
                'f1_macro': float(genus_f1_macro),
                'f1_weighted': float(genus_f1_weighted),
                'num_samples': len(true_genera)
            }
            print(f"Genus (lookup): Acc {genus_accuracy:.4f} ({genus_accuracy*100:.2f}%), F1-macro {genus_f1_macro:.4f}, F1-weighted {genus_f1_weighted:.4f} (N={len(true_genera)})")
    
    # Classification report for species
    report = classification_report(
        eval_results['all_labels'], eval_results['all_preds'], 
        target_names=label_encoder.classes_,
        output_dict=True,
        zero_division=0
    )
    
    # Save report
    report_data = {
        'model_type': 'F_S',
        'species_metrics': {
            'accuracy': float(species_accuracy),
            'f1_macro': float(species_f1_macro),
            'f1_weighted': float(species_f1_weighted)
        },
        'upper_level_metrics': upper_level_metrics,
        'per_class_metrics': report,
        'test_samples': len(true_species_labels),
        'num_classes': len(label_encoder.classes_)
    }
    
    report_filename = f'F_S_evaluation_report.json'
    with open(save_path / report_filename, 'w') as f:
        json.dump(report_data, f, indent=2)
    
    return report

def main():
    parser = argparse.ArgumentParser(description='Evaluate F-S flat model')
    
    # Print separator for model identification
    print("\n" + "=" * 80)
    print("Evaluating Model: F-S")
    print("=" * 80 + "\n")
    
    # Check model file
    if not Path(Config.CHECKPOINT_PATH).exists():
        print(f"Model file not found: {Config.CHECKPOINT_PATH}")
        print(f"Please train the F Species model first")
        print(f"Expected path: outputs/checkpoints/best_F_S_model.pth")
        return
    
    # Load model
    model, label_encoder, config = load_model(Config.CHECKPOINT_PATH)
    
    # Prepare test data
    test_loader, test_df = prepare_test_data(label_encoder)
    
    # Evaluate model
    eval_results = evaluate_model(
        model, test_loader, label_encoder
    )
    
    # Generate report
    report = generate_report(
        eval_results, label_encoder, Config.EVAL_DIR
    )
    
    # Save predictions for error propagation analysis
    # Note: We save species names, and the error propagation script will convert them to IDs
    # using the taxonomy tree and encoders from checkpoints
    def save_predictions(eval_results, label_encoder, test_df, output_path):
        """保存预测结果为JSON（F-S模型：保存species名称，错误传播分析脚本会转换为ID）"""
        # eval_results中的顺序应该与test_loader的顺序一致
        # test_loader的顺序应该与test_df的索引顺序一致（如果shuffle=False）
        true_species_names = eval_results['all_true_labels']
        pred_species_names = eval_results['all_pred_labels']
        filenames = test_df['filename'].astype(str).tolist()
        if len(filenames) != len(true_species_names):
            raise RuntimeError("Prediction count does not match the F-S test manifest")
        
        # 从eval_results获取真实的上层级（已经在evaluate_model中从batch获取）
        true_classes = [str(x) if x else '' for x in eval_results['all_true_classes']]
        true_orders = [str(x) if x else '' for x in eval_results['all_true_orders']]
        true_families = [str(x) if x else '' for x in eval_results['all_true_families']]
        true_genera = [str(x) if x else '' for x in eval_results['all_true_genera']]
        
        # 预测的上层级从eval_results获取（已经通过分类树查找）
        pred_classes = [str(x) if x else '' for x in eval_results['all_pred_classes']]
        pred_orders = [str(x) if x else '' for x in eval_results['all_pred_orders']]
        pred_families = [str(x) if x else '' for x in eval_results['all_pred_families']]
        pred_genera = [str(x) if x else '' for x in eval_results['all_pred_genera']]
        
        # 保存为名称格式（错误传播分析脚本会处理转换）
        predictions_dict = {
            'filename': filenames,
            'true_class': true_classes,
            'true_order': true_orders,
            'true_family': true_families,
            'true_genus': true_genera,
            'true_species': true_species_names,
            'pred_class': pred_classes,
            'pred_order': pred_orders,
            'pred_family': pred_families,
            'pred_genus': pred_genera,
            'pred_species': pred_species_names,
            'format': 'names'  # 标记格式为名称
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(predictions_dict, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Saved predictions: {output_path}")
    
    # Save predictions
    predictions_path = Config.EVAL_DIR / 'F_S_predictions.json'
    save_predictions(eval_results, label_encoder, test_df, predictions_path)

if __name__ == "__main__":
    main()
