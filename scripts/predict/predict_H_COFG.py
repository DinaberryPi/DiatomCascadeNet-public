#!/usr/bin/env python3
"""
DiatomScanNet H-COFG Prediction: Class → Order → Family → Genus
Four-level hierarchical classification prediction script
"""

import os
import sys
from pathlib import Path
import argparse
import json
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import pandas as pd
import numpy as np
import timm

# Add project root to path for utils
from diatom_cascade.prediction import greedy_hierarchical_predict as hierarchical_predict

# Use unified prediction configuration
from diatom_cascade.config.prediction_config import PredictionConfig as Config
from diatom_cascade.runtime import load_checkpoint, get_preprocess_transforms
from diatom_cascade.models import HCOFGModel

# Override checkpoint path for this model
Config.CHECKPOINT_PATH = Config.get_checkpoint_path("H-COFG")


# hierarchical_predict is now imported from utils.hierarchical_predict

class HCOFGPredictor:
    """HCOFG hierarchy predictor for production use"""
    def __init__(self, checkpoint_path=None):
        if checkpoint_path is None:
            checkpoint_path = Config.CHECKPOINT_PATH
        
        self.checkpoint_path = Path(checkpoint_path)
        self.device = Config.DEVICE
        
        # Load model
        print(f"Loading model from: {self.checkpoint_path}")
        # Use unified checkpoint loading
        checkpoint = load_checkpoint(self.checkpoint_path, self.device)
        
        # Extract encoders and matrices
        self.class_encoder = checkpoint.get('class_encoder', None)
        self.order_encoder = checkpoint.get('order_encoder', None)
        self.family_encoder = checkpoint.get('family_encoder', None)
        self.genus_encoder = checkpoint.get('genus_encoder', None)
        self.M_class_order = checkpoint['M_class_order']
        self.M_order_family = checkpoint['M_order_family']
        self.M_family_genus = checkpoint['M_family_genus']
        
        # Create model
        num_classes = checkpoint['num_classes']
        num_orders = checkpoint['num_orders']
        num_families = checkpoint['num_families']
        num_genera = checkpoint.get('num_genera', 0)
        
        self.model = HCOFGModel(
            num_classes,
            num_orders,
            num_families,
            num_genera,
            Config.MODEL_NAME,
            pretrained=False,
        )
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Use unified transforms
        self.transform = get_preprocess_transforms()
        
        print(f"✅ Model loaded successfully")
        print(f"   Classes: {num_classes}")
        print(f"   Orders: {num_orders}")
        print(f"   Families: {num_families}")
        print(f"   Genera: {num_genera}")
    
    def predict_image(self, image_path, topk=2):
        """Predict taxonomic classification for a single image"""
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        original_image = image.copy()
        
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Get model predictions
        with torch.no_grad():
            class_logits, order_logits, family_logits, genus_logits, _, _, _ = self.model(image_tensor)
            
            # Top-down prediction
            pred_class, pred_order, pred_family, pred_genus = hierarchical_predict(
                class_logits, order_logits, 
                family_logits=family_logits, genus_logits=genus_logits,
                M_class_order=self.M_class_order, M_order_family=self.M_order_family, 
                M_family_genus=self.M_family_genus
            )
            
            # Get probabilities for top predictions
            class_probs = torch.softmax(class_logits, dim=1)
            order_probs = torch.softmax(order_logits, dim=1)
            family_probs = torch.softmax(family_logits, dim=1)
            genus_probs = torch.softmax(genus_logits, dim=1)
        
        # Format results
        result = {
            'image_path': str(image_path),
            'predictions': {
                'class': {
                    'name': self.class_encoder.inverse_transform([pred_class.item()])[0],
                    'confidence': float(class_probs[0, pred_class].item())
                },
                'order': {
                    'name': self.order_encoder.inverse_transform([pred_order.item()])[0],
                    'confidence': float(order_probs[0, pred_order].item())
                },
                'family': {
                    'name': self.family_encoder.inverse_transform([pred_family.item()])[0],
                    'confidence': float(family_probs[0, pred_family].item())
                },
                'genus': {
                    'name': self.genus_encoder.inverse_transform([pred_genus.item()])[0],
                    'confidence': float(genus_probs[0, pred_genus].item())
                }
            }
        }
        
        return result
    
    def predict_batch(self, image_paths, topk=2):
        """Predict for multiple images"""
        results = []
        for img_path in image_paths:
            try:
                result = self.predict_image(img_path, topk=topk)
                results.append(result)
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
                results.append({
                    'image_path': str(img_path),
                    'error': str(e)
                })
        return results

def main():
    parser = argparse.ArgumentParser(description='DiatomScanNet H-COFG Hierarchy Prediction')
    
    # Print separator for model identification
    print("\n" + "=" * 80)
    print("Predicting with Model: H-COFG")
    print("=" * 80 + "\n")
    
    parser.add_argument('--image', type=str, help='Path to image file')
    parser.add_argument('--image_dir', type=str, help='Path to directory of images')
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to model checkpoint')
    parser.add_argument('--output', type=str, default='predictions.json', help='Output JSON file')
    parser.add_argument('--output_csv', type=str, help='Output CSV file for batch prediction')
    parser.add_argument('--topk', type=int, default=2, help='Top-k for hierarchical search')
    
    args = parser.parse_args()
    
    # Check if checkpoint exists
    checkpoint_path = args.checkpoint if args.checkpoint else Config.CHECKPOINT_PATH
    if not Path(checkpoint_path).exists():
        print(f"❌ Error: Checkpoint not found at {checkpoint_path}")
        print("Please train the model first using: python train/train_H_COFG.py")
        return
    
    # Initialize predictor
    predictor = HCOFGPredictor(checkpoint_path)
    
    # Collect images
    image_paths = []
    if args.image:
        image_paths = [Path(args.image)]
    elif args.image_dir:
        image_dir = Path(args.image_dir)
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        for ext in image_extensions:
            image_paths.extend(image_dir.glob(f"*{ext}"))
            image_paths.extend(image_dir.glob(f"*{ext.upper()}"))
    else:
        print("❌ Error: Please provide --image or --image_dir")
        return
    
    print(f"\n{'='*70}")
    print(f"Processing {len(image_paths)} image(s)...")
    print(f"{'='*70}\n")
    
    # Make predictions
    results = predictor.predict_batch(image_paths, topk=args.topk)
    
    # Display results
    for result in results:
        if 'error' in result:
            print(f"\n❌ {result['image_path']}: {result['error']}")
            continue
        
        print(f"\n📸 Image: {result['image_path']}")
        print("="*70)
        preds = result['predictions']
        print(f"  Class:  {preds['class']['name']:30s} ({preds['class']['confidence']*100:.1f}%)")
        print(f"  Order:  {preds['order']['name']:30s} ({preds['order']['confidence']*100:.1f}%)")
        print(f"  Family: {preds['family']['name']:30s} ({preds['family']['confidence']*100:.1f}%)")
        print(f"  Genus:  {preds['genus']['name']:30s} ({preds['genus']['confidence']*100:.1f}%)")
    
    # Save results
    output_file = Path(args.output)
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✅ Results saved to: {output_file}")
    
    # Save CSV if requested
    if args.output_csv:
        csv_data = []
        for result in results:
            if 'error' not in result:
                preds = result['predictions']
                csv_data.append({
                    'filename': Path(result['image_path']).name,
                    'class': preds['class']['name'],
                    'class_confidence': preds['class']['confidence'],
                    'order': preds['order']['name'],
                    'order_confidence': preds['order']['confidence'],
                    'family': preds['family']['name'],
                    'family_confidence': preds['family']['confidence'],
                    'genus': preds['genus']['name'],
                    'genus_confidence': preds['genus']['confidence']
                })
            else:
                csv_data.append({
                    'filename': Path(result['image_path']).name,
                    'class': 'ERROR',
                    'class_confidence': 0.0,
                    'order': 'ERROR',
                    'order_confidence': 0.0,
                    'family': 'ERROR',
                    'family_confidence': 0.0,
                    'genus': 'ERROR',
                    'genus_confidence': 0.0
                })
        
        df = pd.DataFrame(csv_data)
        df.to_csv(args.output_csv, index=False)
        print(f"✅ CSV results saved to: {args.output_csv}")

if __name__ == "__main__":
    main()
