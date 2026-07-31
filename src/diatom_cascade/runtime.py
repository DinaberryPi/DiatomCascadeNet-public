"""
Common Utilities

Shared functions used across training, evaluation, and prediction.
These are the core utilities that all three phases need.
"""

import torch
import torchvision.transforms as transforms
import numpy as np
from sklearn.preprocessing import LabelEncoder
from pathlib import Path
from .checkpoints import load_trusted_checkpoint, validate_checkpoint_schema
from .config.train_and_val_config import TrainAndValConfig


def load_checkpoint(checkpoint_path, device=None):
    """
    Load checkpoint with error handling (used by train, eval, and predict)
    
    Args:
        checkpoint_path: Path to checkpoint file
        device: Device to load checkpoint to (if None, uses default from config)
    
    Returns:
        checkpoint dictionary
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if not Path(checkpoint_path).exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    checkpoint = load_trusted_checkpoint(checkpoint_path, map_location=device)
    return validate_checkpoint_schema(checkpoint)


def load_label_encoder(checkpoint, names_key):
    """Rebuild a label encoder from schema-3 plain class names."""
    if names_key not in checkpoint:
        raise ValueError(f"Checkpoint missing '{names_key}'")
    encoder = LabelEncoder()
    encoder.classes_ = np.asarray(checkpoint[names_key])
    return encoder


def get_preprocess_transforms(image_size=None, normalize_mean=None, normalize_std=None):
    """
    Get preprocessing transforms (no augmentation, matches training validation/test)
    Used by evaluation and prediction to ensure consistency.
    
    Args:
        image_size: Image size (defaults to TrainAndValConfig.IMAGE_SIZE)
        normalize_mean: Normalization mean (defaults to TrainAndValConfig.NORMALIZE["mean"])
        normalize_std: Normalization std (defaults to TrainAndValConfig.NORMALIZE["std"])
    
    Returns:
        transforms.Compose
    """
    if image_size is None:
        image_size = TrainAndValConfig.IMAGE_SIZE
    if normalize_mean is None:
        normalize_mean = TrainAndValConfig.NORMALIZE["mean"]
    if normalize_std is None:
        normalize_std = TrainAndValConfig.NORMALIZE["std"]
    
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=normalize_mean, std=normalize_std)
    ])


def get_train_transforms():
    """
    Get training transforms with augmentation (matches TrainAndValConfig)
    
    Returns:
        transforms.Compose
    """
    return TrainAndValConfig.get_train_transforms()


def get_val_test_transforms():
    """
    Get validation/test/prediction transforms (no augmentation, matches TrainAndValConfig)
    Used in validation, evaluation, and prediction phases.
    
    Returns:
        transforms.Compose
    """
    return TrainAndValConfig.get_val_test_transforms()

