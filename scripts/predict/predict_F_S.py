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
import torchvision.transforms as transforms
from PIL import Image
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import timm

# Use unified prediction configuration
from diatom_cascade.config.prediction_config import PredictionConfig as Config
from diatom_cascade.runtime import load_checkpoint, load_label_encoder, get_preprocess_transforms
from diatom_cascade.models import FlatClassifier

# Override checkpoint path for this model
Config.CHECKPOINT_PATH = Config.get_checkpoint_path("F-S")


def load_model(checkpoint_path):
    """Load trained F Species model"""
    print(f"Loading F Species model: {checkpoint_path}")
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Model file not found: {checkpoint_path}")
    
    # Use unified checkpoint loading
    checkpoint = load_checkpoint(checkpoint_path, Config.DEVICE)
    
    label_encoder = load_label_encoder(checkpoint, 'species_names')
    config = checkpoint.get('config', {
        'NUM_CLASSES': checkpoint.get('num_species', len(label_encoder.classes_)),
        'MODEL_NAME': Config.MODEL_NAME,
    })
    
    model = FlatClassifier(
        num_classes=config['NUM_CLASSES'],
        model_name=config.get('MODEL_NAME', Config.MODEL_NAME),
        pretrained=False,
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(Config.DEVICE)
    model.eval()
    
    print(f"✅ F Species model loaded successfully")
    print(f"Number of classes: {config['NUM_CLASSES']}")
    
    return model, label_encoder, config

def preprocess_image(image_path, image_size=None):
    """Preprocess image"""
    if image_size is None:
        image_size = Config.IMAGE_SIZE
    # Use unified transforms from config
    transform = get_preprocess_transforms()
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0)
    return image_tensor, image

def predict_single_image(model, image_path, label_encoder, top_k=3):
    """Predict single image"""
    print(f"\nPredicting: {image_path}")
    
    image_tensor, original_image = preprocess_image(image_path)
    image_tensor = image_tensor.to(Config.DEVICE)
    
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        top_probs, top_indices = torch.topk(probabilities, top_k, dim=1)
    
    top_probs = top_probs.cpu().numpy()[0]
    top_indices = top_indices.cpu().numpy()[0]
    top_classes = [label_encoder.classes_[idx] for idx in top_indices]
    
    print(f"Top {top_k} predictions:")
    for i, (class_name, prob) in enumerate(zip(top_classes, top_probs)):
        print(f"  {i+1}. {class_name}: {prob:.4f} ({prob*100:.2f}%)")
    
    return {
        'image_path': str(image_path),
        'predictions': [{'species': name, 'confidence': float(prob)} for name, prob in zip(top_classes, top_probs)]
    }

def predict_batch(model, image_dir, label_encoder, output_csv=None):
    """Batch prediction"""
    print(f"\nBatch predicting images in: {image_dir}")
    
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    image_files = []
    for ext in image_extensions:
        image_files.extend(Path(image_dir).glob(f"*{ext}"))
        image_files.extend(Path(image_dir).glob(f"*{ext.upper()}"))
    
    if not image_files:
        print("No image files found!")
        return
    
    print(f"Found {len(image_files)} images")
    
    results = []
    for image_path in image_files:
        try:
            image_tensor, _ = preprocess_image(image_path)
            image_tensor = image_tensor.to(Config.DEVICE)
            
            with torch.no_grad():
                outputs = model(image_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                top_prob, top_idx = torch.max(probabilities, dim=1)
            
            predicted_species = label_encoder.classes_[top_idx.item()]
            confidence = top_prob.item()
            
            results.append({
                'filename': image_path.name,
                'predicted_species': predicted_species,
                'confidence': confidence,
                'confidence_pct': confidence * 100
            })
            
            print(f"  {image_path.name}: {predicted_species} ({confidence:.4f})")
            
        except Exception as e:
            print(f"  Error processing {image_path.name}: {e}")
            results.append({
                'filename': image_path.name,
                'predicted_species': 'ERROR',
                'confidence': 0.0,
                'confidence_pct': 0.0
            })
    
    if output_csv:
        df = pd.DataFrame(results)
        df.to_csv(output_csv, index=False)
        print(f"\nResults saved to: {output_csv}")
    
    return results

def main():
    parser = argparse.ArgumentParser(description='DiatomScanNet F Species Baseline Prediction')
    
    # Print separator for model identification
    print("\n" + "=" * 80)
    print("Predicting with Model: F-S")
    print("=" * 80 + "\n")
    
    parser.add_argument('--image', type=str, help='Single image path')
    parser.add_argument('--batch_dir', type=str, help='Image directory path')
    parser.add_argument('--output_csv', type=str, help='Output CSV file path for batch prediction')
    parser.add_argument('--top_k', type=int, default=3, help='Show top K predictions')
    parser.add_argument('--checkpoint', type=str, default=Config.CHECKPOINT_PATH, help='Model checkpoint path')
    
    args = parser.parse_args()
    
    if not Path(args.checkpoint).exists():
        print(f"Model checkpoint not found: {args.checkpoint}")
        print("Please train the model first or specify correct checkpoint path")
        return
    
    model, label_encoder, config = load_model(args.checkpoint)
    
    if args.image:
        if not Path(args.image).exists():
            print(f"Image not found: {args.image}")
            return
        result = predict_single_image(model, args.image, label_encoder, args.top_k)
        print(f"\n✅ Prediction complete!")
        
    elif args.batch_dir:
        if not Path(args.batch_dir).exists():
            print(f"Directory not found: {args.batch_dir}")
            return
        results = predict_batch(model, args.batch_dir, label_encoder, args.output_csv)
        print(f"\n✅ Batch prediction complete! Processed {len(results)} images")
    
    else:
        print("Please specify --image or --batch_dir")
        parser.print_help()

if __name__ == "__main__":
    main()
