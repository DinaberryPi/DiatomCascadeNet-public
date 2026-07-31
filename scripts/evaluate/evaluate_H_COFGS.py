#!/usr/bin/env python3
"""
DiatomScanNet H-COFGS Model Evaluation Script
Evaluate Class→Order→Family→Genus→Species hierarchical model on test set
"""

import os
import sys
import argparse
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, top_k_accuracy_score, f1_score
import timm
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import json

# Add project root to path for utils
from diatom_cascade.prediction import greedy_hierarchical_predict as hierarchical_predict
from diatom_cascade.prediction import level_wise_argmax_predict as argmax_independent_predict

# Use unified evaluation configuration
from diatom_cascade.config.evaluation_config import EvaluationConfig as Config
from diatom_cascade.evaluation import create_test_loader
from diatom_cascade.data.integrity import load_split_manifests, validate_images
from diatom_cascade.runtime import get_preprocess_transforms, load_checkpoint
from diatom_cascade.models import HCOFGSModel

# Override checkpoint path for this model
Config.MODEL_PATH = Config.get_checkpoint_path("H-COFGS")

Config.EVAL_DIR.mkdir(parents=True, exist_ok=True)


class HCOFGSDataset(Dataset):
    """Dataset for H-COFGS model"""
    def __init__(self, df, class_encoder, order_encoder, family_encoder, genus_encoder, species_encoder, transform=None):
        self.df = df.reset_index(drop=True)
        self.class_encoder = class_encoder
        self.order_encoder = order_encoder
        self.family_encoder = family_encoder
        self.genus_encoder = genus_encoder
        self.species_encoder = species_encoder
        self.transform = transform
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_path = Config.IMAGES_DIR / row['filename']
        
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        
        class_label = int(self.class_encoder.transform([row['class']])[0])
        order_label = int(self.order_encoder.transform([row['order']])[0])
        family_label = int(self.family_encoder.transform([row['family']])[0])
        genus_label = int(self.genus_encoder.transform([row['genus']])[0])
        species_label = int(self.species_encoder.transform([row['species']])[0])
        
        return image, class_label, order_label, family_label, genus_label, species_label

# hierarchical_predict is now imported from utils.hierarchical_predict
from diatom_cascade.prediction import beam_search_hierarchical_predict

def load_and_prepare_data():
    """Load and prepare data - use pre-filtered dataset"""
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
    
    model_type = "H-COFGS"
    if model_type not in model_data_mapping:
        raise ValueError(f"Model type '{model_type}' not found in mapping")
    
    # Load the correct pre-filtered dataset file
    # Handle relative paths (e.g., "../cleaned/labels_clean.csv" for F-C)
    dataset_path = model_data_mapping[model_type]
    dataset_file = (Config.DATA_ROOT / "preprocessed" / dataset_path).resolve()
    Config.LABELS_CSV = dataset_file
    
    # Always show dataset info (even in quiet mode)
    print(f"Model: {model_type}")
    print(f"Dataset: {dataset_file.name}")
    
    df = pd.read_csv(Config.LABELS_CSV)
    
    class_encoder = LabelEncoder()
    class_encoder.fit(df['class'].unique())
    
    order_encoder = LabelEncoder()
    order_encoder.fit(df['order'].unique())
    
    family_encoder = LabelEncoder()
    family_encoder.fit(df['family'].unique())
    
    genus_encoder = LabelEncoder()
    genus_encoder.fit(df['genus'].unique())
    
    species_encoder = LabelEncoder()
    species_encoder.fit(df['species'].unique())
    
    return df, class_encoder, order_encoder, family_encoder, genus_encoder, species_encoder

def main():
    parser = argparse.ArgumentParser(description='Evaluate H-COFGS hierarchical model')
    
    # Print separator for model identification
    print("\n" + "=" * 80)
    print("Evaluating Model: H-COFGS")
    print("=" * 80 + "\n")
    
    # Load data
    df, class_encoder, order_encoder, family_encoder, genus_encoder, species_encoder = load_and_prepare_data()
    
    _, _, test_df = load_split_manifests(Config.DATA_ROOT, "H-COFGS")
    validate_images(test_df, Config.IMAGES_DIR,
                    Config.OUTPUT_DIR / "preflight" / "H-COFGS_test_images.csv")
    
    # Data transforms (use unified config)
    test_transforms = get_preprocess_transforms()
    
    # Create test dataset
    test_dataset = HCOFGSDataset(test_df, class_encoder, order_encoder, family_encoder, genus_encoder, species_encoder, test_transforms)
    test_loader = create_test_loader(test_dataset)
    
    # Load model (use unified checkpoint loading)
    checkpoint = load_checkpoint(Config.MODEL_PATH, Config.DEVICE)
    
    num_classes = checkpoint['num_classes']
    num_orders = checkpoint['num_orders']
    num_families = checkpoint['num_families']
    num_genera = checkpoint['num_genera']
    num_species = checkpoint['num_species']
    M_class_order = checkpoint['M_class_order']
    M_order_family = checkpoint['M_order_family']
    M_family_genus = checkpoint['M_family_genus']
    M_genus_species = checkpoint['M_genus_species']
    
    model = HCOFGSModel(
        num_classes,
        num_orders,
        num_families,
        num_genera,
        num_species,
        Config.MODEL_NAME,
        pretrained=False,
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(Config.DEVICE)
    model.eval()
    print(f"✅ Model loaded ({num_classes} classes, {num_orders} orders, {num_families} families, {num_genera} genera, {num_species} species)")
    print(f"Test set: {len(test_df)} samples")
    
    # Greedy hierarchical prediction results (with mask constraints)
    all_class_true = []
    all_class_pred_hier = []
    all_order_true = []
    all_order_pred_hier = []
    all_family_true = []
    all_family_pred_hier = []
    all_genus_true = []
    all_genus_pred_hier = []
    all_species_true = []
    all_species_pred_hier = []
    
    # Level-wise argmax prediction results (no mask constraints, for comparison only)
    all_class_pred_argmax = []
    all_order_pred_argmax = []
    all_family_pred_argmax = []
    all_genus_pred_argmax = []
    all_species_pred_argmax = []
    
    # Beam search prediction results (globally constrained)
    all_class_pred_beam = []
    all_order_pred_beam = []
    all_family_pred_beam = []
    all_genus_pred_beam = []
    all_species_pred_beam = []
    
    # Confidence scores (max probability for each level)
    all_class_conf_hier = []
    all_order_conf_hier = []
    all_family_conf_hier = []
    all_genus_conf_hier = []
    all_species_conf_hier = []
    
    all_class_conf_argmax = []
    all_order_conf_argmax = []
    all_family_conf_argmax = []
    all_genus_conf_argmax = []
    all_species_conf_argmax = []
    
    with torch.no_grad():
        for images, class_labels, order_labels, family_labels, genus_labels, species_labels in tqdm(test_loader, desc="Evaluating"):
            images = images.to(Config.DEVICE)
            
            class_logits, order_logits, family_logits, genus_logits, species_logits, _, _, _, _ = model(images)
            
            # Convert to probabilities for confidence scores
            class_probs = torch.softmax(class_logits, dim=1)
            order_probs = torch.softmax(order_logits, dim=1)
            family_probs = torch.softmax(family_logits, dim=1)
            genus_probs = torch.softmax(genus_logits, dim=1)
            species_probs = torch.softmax(species_logits, dim=1)
            
            # Method 1: Greedy hierarchical prediction (with mask constraints)
            pred_class_hier, pred_order_hier, pred_family_hier, pred_genus_hier, pred_species_hier = hierarchical_predict(
                class_logits, order_logits, 
                family_logits=family_logits, genus_logits=genus_logits, species_logits=species_logits,
                M_class_order=M_class_order, M_order_family=M_order_family, 
                M_family_genus=M_family_genus, M_genus_species=M_genus_species
            )
            
            # Method 2: Level-wise argmax (no mask constraints, for comparison only)
            pred_class_argmax, pred_order_argmax, pred_family_argmax, pred_genus_argmax, pred_species_argmax = argmax_independent_predict(
                class_logits, order_logits, family_logits=family_logits, genus_logits=genus_logits, species_logits=species_logits
            )
            
            # Method 3: Beam search prediction (globally constrained)
            pred_class_beam, pred_order_beam, pred_family_beam, pred_genus_beam, pred_species_beam = beam_search_hierarchical_predict(
                class_logits, order_logits,
                family_logits=family_logits, genus_logits=genus_logits, species_logits=species_logits,
                M_class_order=M_class_order, M_order_family=M_order_family,
                M_family_genus=M_family_genus, M_genus_species=M_genus_species,
                beam_width=Config.BEAM_WIDTH
            )
            
            # Collect greedy hierarchical predictions (with mask constraints)
            all_class_true.extend(class_labels.cpu().numpy())
            all_class_pred_hier.extend(pred_class_hier.cpu().numpy())
            all_order_true.extend(order_labels.cpu().numpy())
            all_order_pred_hier.extend(pred_order_hier.cpu().numpy())
            all_family_true.extend(family_labels.cpu().numpy())
            all_family_pred_hier.extend(pred_family_hier.cpu().numpy())
            all_genus_true.extend(genus_labels.cpu().numpy())
            all_genus_pred_hier.extend(pred_genus_hier.cpu().numpy())
            all_species_true.extend(species_labels.cpu().numpy())
            all_species_pred_hier.extend(pred_species_hier.cpu().numpy())
            
            # Collect argmax predictions
            all_class_pred_argmax.extend(pred_class_argmax.cpu().numpy())
            all_order_pred_argmax.extend(pred_order_argmax.cpu().numpy())
            all_family_pred_argmax.extend(pred_family_argmax.cpu().numpy())
            all_genus_pred_argmax.extend(pred_genus_argmax.cpu().numpy())
            all_species_pred_argmax.extend(pred_species_argmax.cpu().numpy())
            
            # Collect beam search predictions
            all_class_pred_beam.extend(pred_class_beam.cpu().numpy())
            all_order_pred_beam.extend(pred_order_beam.cpu().numpy())
            all_family_pred_beam.extend(pred_family_beam.cpu().numpy())
            all_genus_pred_beam.extend(pred_genus_beam.cpu().numpy())
            all_species_pred_beam.extend(pred_species_beam.cpu().numpy())
            
            # Collect confidence scores (max probability)
            batch_size = class_logits.shape[0]
            for b in range(batch_size):
                # Hierarchical confidence (probability of predicted class)
                all_class_conf_hier.append(class_probs[b, pred_class_hier[b]].item())
                all_order_conf_hier.append(order_probs[b, pred_order_hier[b]].item())
                all_family_conf_hier.append(family_probs[b, pred_family_hier[b]].item())
                all_genus_conf_hier.append(genus_probs[b, pred_genus_hier[b]].item())
                all_species_conf_hier.append(species_probs[b, pred_species_hier[b]].item())
                
                # Argmax confidence (max probability)
                all_class_conf_argmax.append(class_probs[b, pred_class_argmax[b]].item())
                all_order_conf_argmax.append(order_probs[b, pred_order_argmax[b]].item())
                all_family_conf_argmax.append(family_probs[b, pred_family_argmax[b]].item())
                all_genus_conf_argmax.append(genus_probs[b, pred_genus_argmax[b]].item())
                all_species_conf_argmax.append(species_probs[b, pred_species_argmax[b]].item())
    
    # Calculate metrics for greedy hierarchical prediction (with mask constraints)
    class_acc_hier = accuracy_score(all_class_true, all_class_pred_hier)
    class_f1_hier = f1_score(all_class_true, all_class_pred_hier, average='weighted', zero_division=0)
    order_acc_hier = accuracy_score(all_order_true, all_order_pred_hier)
    order_f1_hier = f1_score(all_order_true, all_order_pred_hier, average='weighted', zero_division=0)
    family_acc_hier = accuracy_score(all_family_true, all_family_pred_hier)
    family_f1_hier = f1_score(all_family_true, all_family_pred_hier, average='weighted', zero_division=0)
    genus_acc_hier = accuracy_score(all_genus_true, all_genus_pred_hier)
    genus_f1_hier = f1_score(all_genus_true, all_genus_pred_hier, average='weighted', zero_division=0)
    species_acc_hier = accuracy_score(all_species_true, all_species_pred_hier)
    species_f1_hier = f1_score(all_species_true, all_species_pred_hier, average='weighted', zero_division=0)
    
    # Calculate metrics for level-wise argmax prediction (no mask constraints)
    class_acc_argmax = accuracy_score(all_class_true, all_class_pred_argmax)
    class_f1_argmax = f1_score(all_class_true, all_class_pred_argmax, average='weighted', zero_division=0)
    order_acc_argmax = accuracy_score(all_order_true, all_order_pred_argmax)
    order_f1_argmax = f1_score(all_order_true, all_order_pred_argmax, average='weighted', zero_division=0)
    family_acc_argmax = accuracy_score(all_family_true, all_family_pred_argmax)
    family_f1_argmax = f1_score(all_family_true, all_family_pred_argmax, average='weighted', zero_division=0)
    genus_acc_argmax = accuracy_score(all_genus_true, all_genus_pred_argmax)
    genus_f1_argmax = f1_score(all_genus_true, all_genus_pred_argmax, average='weighted', zero_division=0)
    species_acc_argmax = accuracy_score(all_species_true, all_species_pred_argmax)
    species_f1_argmax = f1_score(all_species_true, all_species_pred_argmax, average='weighted', zero_division=0)
    
    # Calculate metrics for beam search prediction
    class_acc_beam = accuracy_score(all_class_true, all_class_pred_beam)
    class_f1_beam = f1_score(all_class_true, all_class_pred_beam, average='weighted', zero_division=0)
    order_acc_beam = accuracy_score(all_order_true, all_order_pred_beam)
    order_f1_beam = f1_score(all_order_true, all_order_pred_beam, average='weighted', zero_division=0)
    family_acc_beam = accuracy_score(all_family_true, all_family_pred_beam)
    family_f1_beam = f1_score(all_family_true, all_family_pred_beam, average='weighted', zero_division=0)
    genus_acc_beam = accuracy_score(all_genus_true, all_genus_pred_beam)
    genus_f1_beam = f1_score(all_genus_true, all_genus_pred_beam, average='weighted', zero_division=0)
    species_acc_beam = accuracy_score(all_species_true, all_species_pred_beam)
    species_f1_beam = f1_score(all_species_true, all_species_pred_beam, average='weighted', zero_division=0)
    
    # Calculate average confidence scores
    avg_class_conf_hier = np.mean(all_class_conf_hier)
    avg_order_conf_hier = np.mean(all_order_conf_hier)
    avg_family_conf_hier = np.mean(all_family_conf_hier)
    avg_genus_conf_hier = np.mean(all_genus_conf_hier)
    avg_species_conf_hier = np.mean(all_species_conf_hier)
    
    avg_class_conf_argmax = np.mean(all_class_conf_argmax)
    avg_order_conf_argmax = np.mean(all_order_conf_argmax)
    avg_family_conf_argmax = np.mean(all_family_conf_argmax)
    avg_genus_conf_argmax = np.mean(all_genus_conf_argmax)
    avg_species_conf_argmax = np.mean(all_species_conf_argmax)
    
    # Calculate agreement rates
    class_agreement_hier_argmax = np.mean(np.array(all_class_pred_hier) == np.array(all_class_pred_argmax))
    order_agreement_hier_argmax = np.mean(np.array(all_order_pred_hier) == np.array(all_order_pred_argmax))
    family_agreement_hier_argmax = np.mean(np.array(all_family_pred_hier) == np.array(all_family_pred_argmax))
    genus_agreement_hier_argmax = np.mean(np.array(all_genus_pred_hier) == np.array(all_genus_pred_argmax))
    species_agreement_hier_argmax = np.mean(np.array(all_species_pred_hier) == np.array(all_species_pred_argmax))
    
    class_agreement_hier_beam = np.mean(np.array(all_class_pred_hier) == np.array(all_class_pred_beam))
    order_agreement_hier_beam = np.mean(np.array(all_order_pred_hier) == np.array(all_order_pred_beam))
    family_agreement_hier_beam = np.mean(np.array(all_family_pred_hier) == np.array(all_family_pred_beam))
    genus_agreement_hier_beam = np.mean(np.array(all_genus_pred_hier) == np.array(all_genus_pred_beam))
    species_agreement_hier_beam = np.mean(np.array(all_species_pred_hier) == np.array(all_species_pred_beam))
    
    # Calculate agreement rates
    class_agreement_hier_argmax = np.mean(np.array(all_class_pred_hier) == np.array(all_class_pred_argmax))
    order_agreement_hier_argmax = np.mean(np.array(all_order_pred_hier) == np.array(all_order_pred_argmax))
    family_agreement_hier_argmax = np.mean(np.array(all_family_pred_hier) == np.array(all_family_pred_argmax))
    genus_agreement_hier_argmax = np.mean(np.array(all_genus_pred_hier) == np.array(all_genus_pred_argmax))
    species_agreement_hier_argmax = np.mean(np.array(all_species_pred_hier) == np.array(all_species_pred_argmax))
    
    class_agreement_hier_beam = np.mean(np.array(all_class_pred_hier) == np.array(all_class_pred_beam))
    order_agreement_hier_beam = np.mean(np.array(all_order_pred_hier) == np.array(all_order_pred_beam))
    family_agreement_hier_beam = np.mean(np.array(all_family_pred_hier) == np.array(all_family_pred_beam))
    genus_agreement_hier_beam = np.mean(np.array(all_genus_pred_hier) == np.array(all_genus_pred_beam))
    species_agreement_hier_beam = np.mean(np.array(all_species_pred_hier) == np.array(all_species_pred_beam))
    
    print(f"Greedy: C {class_acc_hier:.4f} | O {order_acc_hier:.4f} | F {family_acc_hier:.4f} | G {genus_acc_hier:.4f} | S {species_acc_hier:.4f} (N={len(test_df)})")
    print(f"Argmax: C {class_acc_argmax:.4f} | O {order_acc_argmax:.4f} | F {family_acc_argmax:.4f} | G {genus_acc_argmax:.4f} | S {species_acc_argmax:.4f} (N={len(test_df)})")
    print(f"Beam:   C {class_acc_beam:.4f} | O {order_acc_beam:.4f} | F {family_acc_beam:.4f} | G {genus_acc_beam:.4f} | S {species_acc_beam:.4f} (N={len(test_df)})")
    
    # Save comprehensive report with both methods
    report = {
        'hierarchical_prediction': {
            'class_level': {
                'accuracy': float(class_acc_hier),
                'f1_weighted': float(class_f1_hier),
                'avg_confidence': float(avg_class_conf_hier)
            },
            'order_level': {
                'accuracy': float(order_acc_hier),
                'f1_weighted': float(order_f1_hier),
                'avg_confidence': float(avg_order_conf_hier)
            },
            'family_level': {
                'accuracy': float(family_acc_hier),
                'f1_weighted': float(family_f1_hier),
                'avg_confidence': float(avg_family_conf_hier)
            },
            'genus_level': {
                'accuracy': float(genus_acc_hier),
                'f1_weighted': float(genus_f1_hier),
                'avg_confidence': float(avg_genus_conf_hier)
            },
            'species_level': {
                'accuracy': float(species_acc_hier),
                'f1_weighted': float(species_f1_hier),
                'avg_confidence': float(avg_species_conf_hier),
                'num_species': int(num_species)
            }
        },
        'argmax_prediction': {
            'class_level': {
                'accuracy': float(class_acc_argmax),
                'f1_weighted': float(class_f1_argmax),
                'avg_confidence': float(avg_class_conf_argmax)
            },
            'order_level': {
                'accuracy': float(order_acc_argmax),
                'f1_weighted': float(order_f1_argmax),
                'avg_confidence': float(avg_order_conf_argmax)
            },
            'family_level': {
                'accuracy': float(family_acc_argmax),
                'f1_weighted': float(family_f1_argmax),
                'avg_confidence': float(avg_family_conf_argmax)
            },
            'genus_level': {
                'accuracy': float(genus_acc_argmax),
                'f1_weighted': float(genus_f1_argmax),
                'avg_confidence': float(avg_genus_conf_argmax)
            },
            'species_level': {
                'accuracy': float(species_acc_argmax),
                'f1_weighted': float(species_f1_argmax),
                'avg_confidence': float(avg_species_conf_argmax),
                'num_species': int(num_species)
            }
        },
        'beam_search_prediction': {
            'class_level': {
                'accuracy': float(class_acc_beam),
                'f1_weighted': float(class_f1_beam),
                'avg_confidence': float(avg_class_conf_hier)  # Use hierarchical confidence for beam
            },
            'order_level': {
                'accuracy': float(order_acc_beam),
                'f1_weighted': float(order_f1_beam),
                'avg_confidence': float(avg_order_conf_hier)
            },
            'family_level': {
                'accuracy': float(family_acc_beam),
                'f1_weighted': float(family_f1_beam),
                'avg_confidence': float(avg_family_conf_hier)
            },
            'genus_level': {
                'accuracy': float(genus_acc_beam),
                'f1_weighted': float(genus_f1_beam),
                'avg_confidence': float(avg_genus_conf_hier)
            },
            'species_level': {
                'accuracy': float(species_acc_beam),
                'f1_weighted': float(species_f1_beam),
                'avg_confidence': float(avg_species_conf_hier),
                'num_species': int(num_species)
            }
        },
        'comparison': {
            'agreement_rate_hier_argmax': {
                'class': float(class_agreement_hier_argmax),
                'order': float(order_agreement_hier_argmax),
                'family': float(family_agreement_hier_argmax),
                'genus': float(genus_agreement_hier_argmax),
                'species': float(species_agreement_hier_argmax)
            },
            'agreement_rate_hier_beam': {
                'class': float(class_agreement_hier_beam),
                'order': float(order_agreement_hier_beam),
                'family': float(family_agreement_hier_beam),
                'genus': float(genus_agreement_hier_beam),
                'species': float(species_agreement_hier_beam)
            },
            'accuracy_difference_hier_argmax': {
                'class': float(class_acc_hier - class_acc_argmax),
                'order': float(order_acc_hier - order_acc_argmax),
                'family': float(family_acc_hier - family_acc_argmax),
                'genus': float(genus_acc_hier - genus_acc_argmax),
                'species': float(species_acc_hier - species_acc_argmax)
            },
            'accuracy_difference_hier_beam': {
                'class': float(class_acc_hier - class_acc_beam),
                'order': float(order_acc_hier - order_acc_beam),
                'family': float(family_acc_hier - family_acc_beam),
                'genus': float(genus_acc_hier - genus_acc_beam),
                'species': float(species_acc_hier - species_acc_beam)
            },
            'accuracy_difference_beam_argmax': {
                'class': float(class_acc_beam - class_acc_argmax),
                'order': float(order_acc_beam - order_acc_argmax),
                'family': float(family_acc_beam - family_acc_argmax),
                'genus': float(genus_acc_beam - genus_acc_argmax),
                'species': float(species_acc_beam - species_acc_argmax)
            },
            'confidence_difference': {
                'class': float(avg_class_conf_hier - avg_class_conf_argmax),
                'order': float(avg_order_conf_hier - avg_order_conf_argmax),
                'family': float(avg_family_conf_hier - avg_family_conf_argmax),
                'genus': float(avg_genus_conf_hier - avg_genus_conf_argmax),
                'species': float(avg_species_conf_hier - avg_species_conf_argmax)
            }
        },
        'test_samples': len(test_df)
    }
    
    report_path = Config.EVAL_DIR / 'H_COFGS_evaluation_report.json'
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    # Save predictions for error propagation analysis (greedy method)
    def save_predictions(all_predictions, test_df, output_path):
        """保存预测结果为JSON"""
        # 转换为可序列化的格式
        filenames = test_df['filename'].astype(str).tolist()
        if len(filenames) != len(all_predictions['true_species']):
            raise RuntimeError("Prediction count does not match the H-COFGS test manifest")
        predictions_dict = {
            'filename': filenames,
            'true_class': [int(x) for x in all_predictions['true_class']],
            'true_order': [int(x) for x in all_predictions['true_order']],
            'true_family': [int(x) for x in all_predictions['true_family']],
            'true_genus': [int(x) for x in all_predictions['true_genus']],
            'true_species': [int(x) for x in all_predictions['true_species']],
            'pred_class': [int(x) for x in all_predictions['pred_class']],
            'pred_order': [int(x) for x in all_predictions['pred_order']],
            'pred_family': [int(x) for x in all_predictions['pred_family']],
            'pred_genus': [int(x) for x in all_predictions['pred_genus']],
            'pred_species': [int(x) for x in all_predictions['pred_species']],
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(predictions_dict, f, indent=2)
        
        print(f"✅ Saved predictions: {output_path}")
    
    # Save greedy predictions for error propagation analysis
    all_predictions_greedy = {
        'true_class': all_class_true,
        'true_order': all_order_true,
        'true_family': all_family_true,
        'true_genus': all_genus_true,
        'true_species': all_species_true,
        'pred_class': all_class_pred_hier,
        'pred_order': all_order_pred_hier,
        'pred_family': all_family_pred_hier,
        'pred_genus': all_genus_pred_hier,
        'pred_species': all_species_pred_hier,
    }
    
    predictions_path = Config.EVAL_DIR / 'H_COFGS_greedy_predictions.json'
    save_predictions(all_predictions_greedy, test_df, predictions_path)

if __name__ == "__main__":
    main()
