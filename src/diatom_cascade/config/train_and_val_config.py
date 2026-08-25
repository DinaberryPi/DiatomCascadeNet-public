"""
Standardized Training and Validation Configuration for DiatomScanNet
Single source of truth for all model training and validation hyperparameters
"""

from pathlib import Path
import random
import numpy as np
import torch
import torchvision.transforms as transforms
from . import split_config
from .data_config import IMAGE_SIZE as DATA_IMAGE_SIZE
from .data_config import MIN_SAMPLES as DATA_MIN_SAMPLES
from .model_config import (
    BACKBONE_PRETRAIN as CONFIGURED_BACKBONE_PRETRAIN,
    BASE_MODEL as CONFIGURED_BASE_MODEL,
    HEAD_SPECS,
)


class TrainAndValConfig:
    """
    Standardized training and validation configuration for all models.
    Includes both training (with augmentation) and validation (without augmentation) settings.
    
    Category 1: Architecture-Independent Parameters (MUST BE CONSISTENT ACROSS ALL MODELS)
    Category 2: Architecture-Dependent Parameters (UNIFIED START)
    Category 3: Task-Specific Parameters (CAN BE DIFFERENT)
    """
    
    # ============================================================================
    # Category 1: Architecture-Independent Parameters (MUST BE SAME)
    # ============================================================================
    
    # Data & Model Architecture
    IMAGE_SIZE = DATA_IMAGE_SIZE
    BATCH_SIZE = 32
    BASE_MODEL = CONFIGURED_BASE_MODEL
    BACKBONE_PRETRAIN = CONFIGURED_BACKBONE_PRETRAIN
    
    # Optimizer
    OPTIMIZER = "AdamW"
    WEIGHT_DECAY = 1e-4
    
    # Training
    MAX_EPOCHS = 80  # Changed from 50
    RANDOM_SEED = split_config.RANDOM_SEED
    
    # DataLoader
    NUM_WORKERS = 4
    PIN_MEMORY = True
    
    # ============================================================================
    # Category 2: Architecture-Dependent Parameters (UNIFIED START)
    # ============================================================================
    
    # Learning Rate (UNIFIED START)
    INITIAL_LR = 5e-4  # 0.0005 - unified starting point for all models
    
    # Learning Rate Scheduler (UNIFIED)
    LR_SCHEDULER_TYPE = "ReduceLROnPlateau"  # Changed from CosineAnnealingLR
    LR_SCHEDULER_CONFIG = {
        "mode": "min",           # Monitor validation loss
        "factor": 0.5,           # LR *= 0.5 when plateau
        "patience": 5,           # Wait 5 epochs before reducing
        "min_lr": 1e-6          # Minimum learning rate
    }
    
    # Early Stopping (UNIFIED)
    EARLY_STOPPING_PATIENCE = 15  # Changed from 10
    EARLY_STOPPING_MIN_DELTA = 0.001
    EARLY_STOPPING_MODE = "max"  # Monitor F1 score (maximize)
    
    # ============================================================================
    # Category 3: Task-Specific Parameters (CAN BE DIFFERENT)
    # ============================================================================
    
    # Data Split (UNIFIED RATIO)
    TRAIN_RATIO = split_config.TRAIN_RATIO
    VAL_RATIO = split_config.VAL_RATIO
    TEST_RATIO = split_config.TEST_RATIO
    
    # Stratification (different per model to ensure balance)
    STRATIFY_BY = split_config.STRATIFY_BY
    
    # Loss Function: All models use Focal Loss (α=0.25, γ=2.0) for fair comparison
    # Flat models: Standard Focal Loss
    # Hierarchical models: Masked Focal Loss with weighted combination
    FOCAL_ALPHA = 0.25
    FOCAL_GAMMA = 2.0
    
    # Loss Weights (Hierarchical Models Only)
    LOSS_WEIGHTS = {
        "H-CO": {
            "class": 1.0,
            "order": 1.0
        },
        "H-COF": {
            "class": 1.0,
            "order": 1.0,
            "family": 1.0
        },
        "H-COFG": {
            "class": 0.8,
            "order": 0.9,
            "family": 1.0,
            "genus": 1.2
        },
        "H-COFGS": {
            "class": 0.8,
            "order": 0.9,
            "family": 1.0,
            "genus": 1.2,
            "species": 1.5
        }
    }
    
    # Dropout (Model-Specific - Keep Current Design)
    DROPOUT_FLAT = HEAD_SPECS["flat"][1]
    DROPOUT_HEAD_2LAYER = HEAD_SPECS["class"][1]
    DROPOUT_HEAD_3LAYER = HEAD_SPECS["order"][1]
    DROPOUT_HEAD_4LAYER = HEAD_SPECS["genus"][1]
    
    # Minimum Sample Filtering
    # All levels use MIN_SAMPLES=10 for consistency and data quality
    MIN_SAMPLES = DATA_MIN_SAMPLES
    
    # ============================================================================
    # Data Augmentation & Normalization
    # ============================================================================
    
    # Data Augmentation (Training only)
    AUGMENTATION = {
        "RandomHorizontalFlip": {"p": 0.5},
        "RandomRotation": {"degrees": 15},
        "ColorJitter": {
            "brightness": 0.2,
            "contrast": 0.2,
            "saturation": 0.2,
            "hue": 0.1
        }
    }
    
    # Normalization (Train + Val + Test)
    NORMALIZE = {
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225]
    }
    
    # ============================================================================
    # Helper Methods
    # ============================================================================
    
    @staticmethod
    def set_global_seed(seed=None):
        """Seed Python, NumPy, PyTorch, and cuDNN from the shared config."""
        if seed is None:
            seed = TrainAndValConfig.RANDOM_SEED
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    @staticmethod
    def get_train_transforms():
        """Get training transforms with augmentation"""
        return transforms.Compose([
            transforms.Resize((TrainAndValConfig.IMAGE_SIZE, TrainAndValConfig.IMAGE_SIZE)),
            transforms.RandomHorizontalFlip(p=TrainAndValConfig.AUGMENTATION["RandomHorizontalFlip"]["p"]),
            transforms.RandomRotation(degrees=TrainAndValConfig.AUGMENTATION["RandomRotation"]["degrees"]),
            transforms.ColorJitter(
                brightness=TrainAndValConfig.AUGMENTATION["ColorJitter"]["brightness"],
                contrast=TrainAndValConfig.AUGMENTATION["ColorJitter"]["contrast"],
                saturation=TrainAndValConfig.AUGMENTATION["ColorJitter"]["saturation"],
                hue=TrainAndValConfig.AUGMENTATION["ColorJitter"]["hue"]
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=TrainAndValConfig.NORMALIZE["mean"],
                std=TrainAndValConfig.NORMALIZE["std"]
            )
        ])
    
    @staticmethod
    def get_val_test_transforms():
        """Get validation/test transforms (no augmentation)"""
        return transforms.Compose([
            transforms.Resize((TrainAndValConfig.IMAGE_SIZE, TrainAndValConfig.IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=TrainAndValConfig.NORMALIZE["mean"],
                std=TrainAndValConfig.NORMALIZE["std"]
            )
        ])
    
    @staticmethod
    def get_lr_scheduler(optimizer):
        """Get learning rate scheduler based on config"""
        if TrainAndValConfig.LR_SCHEDULER_TYPE == "ReduceLROnPlateau":
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                **TrainAndValConfig.LR_SCHEDULER_CONFIG
            )
        elif TrainAndValConfig.LR_SCHEDULER_TYPE == "CosineAnnealingLR":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=TrainAndValConfig.MAX_EPOCHS
            )
        else:
            raise ValueError(f"Unknown scheduler type: {TrainAndValConfig.LR_SCHEDULER_TYPE}")

