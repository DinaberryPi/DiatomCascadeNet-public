"""
Standardized Prediction Configuration for DiatomScanNet
Single source of truth for all model prediction parameters
All parameters must match TrainingConfig and EvaluationConfig to ensure consistency
"""

from pathlib import Path
import torch
from .evaluation_config import EvaluationConfig
from .path_config import get_output_dir
from .train_and_val_config import TrainAndValConfig


class PredictionConfig:
    """
    Standardized prediction configuration for all models.
    All parameters must match TrainingConfig and EvaluationConfig to ensure consistency.
    """
    
    # ============================================================================
    # Model & Data Configuration (MUST MATCH TRAINING/EVALUATION)
    # ============================================================================
    
    # Model Architecture (MUST MATCH TRAINING)
    MODEL_NAME = TrainAndValConfig.BASE_MODEL
    IMAGE_SIZE = TrainAndValConfig.IMAGE_SIZE
    
    # DataLoader (MUST MATCH TRAINING/EVALUATION)
    BATCH_SIZE = TrainAndValConfig.BATCH_SIZE
    
    # Device
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # ============================================================================
    # Data Paths
    # ============================================================================
    
    DATA_ROOT = Path("dataset")
    IMAGES_DIR = DATA_ROOT / "raw" / "images"
    TAXONOMY_JSON = DATA_ROOT / "preprocessed" / "taxonomy_tree.json"
    
    # ============================================================================
    # Output Paths
    # ============================================================================
    
    OUTPUT_DIR = get_output_dir()
    CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
    PREDICTION_DIR = OUTPUT_DIR / "predictions"
    PREDICTION_DIR.mkdir(parents=True, exist_ok=True)
    
    # ============================================================================
    # Model Checkpoint Paths
    # ============================================================================
    
    CHECKPOINT_PATHS = {
        "F-C": CHECKPOINT_DIR / "best_F_C_model.pth",
        "F-G": CHECKPOINT_DIR / "best_F_G_model.pth",
        "F-S": CHECKPOINT_DIR / "best_F_S_model.pth",
        "H-CO": CHECKPOINT_DIR / "best_H_CO_model.pth",
        "H-COF": CHECKPOINT_DIR / "best_H_COF_model.pth",
        "H-COFG": CHECKPOINT_DIR / "best_H_COFG_model.pth",
        "H-COFGS": CHECKPOINT_DIR / "best_H_COFGS_model.pth"
    }
    
    # ============================================================================
    # Prediction Parameters
    # ============================================================================
    
    # Beam Search (for hierarchical models)
    BEAM_WIDTH = 3
    
    # Top-K predictions
    TOP_K = 5
    
    # Verbose output
    VERBOSE = True
    
    @staticmethod
    def get_checkpoint_path(model_name):
        """Get checkpoint path for a model"""
        return PredictionConfig.CHECKPOINT_PATHS.get(model_name)
    
    @staticmethod
    def get_preprocess_transforms():
        """Get preprocessing transforms (matches training/evaluation)"""
        import torchvision.transforms as transforms
        return transforms.Compose([
            transforms.Resize((PredictionConfig.IMAGE_SIZE, PredictionConfig.IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=TrainAndValConfig.NORMALIZE["mean"],
                std=TrainAndValConfig.NORMALIZE["std"]
            )
        ])

