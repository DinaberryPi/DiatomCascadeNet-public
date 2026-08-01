"""
Standardized Evaluation Configuration for DiatomScanNet
Single source of truth for all model evaluation parameters
All parameters must match TrainingConfig to ensure fair comparison
"""

from pathlib import Path
import torch
from .path_config import get_data_root, get_output_dir
from .train_and_val_config import TrainAndValConfig


class EvaluationConfig:
    """
    Standardized evaluation configuration for all models.
    All parameters must match TrainingConfig to ensure fair comparison.
    """
    
    # ============================================================================
    # Model & Data Configuration (MUST MATCH TRAINING)
    # ============================================================================
    
    # Model Architecture (MUST MATCH TRAINING)
    MODEL_NAME = TrainAndValConfig.BASE_MODEL
    IMAGE_SIZE = TrainAndValConfig.IMAGE_SIZE
    
    # DataLoader (MUST MATCH TRAINING)
    BATCH_SIZE = TrainAndValConfig.BATCH_SIZE
    NUM_WORKERS = TrainAndValConfig.NUM_WORKERS
    PIN_MEMORY = TrainAndValConfig.PIN_MEMORY
    
    # Device
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # ============================================================================
    # Data Paths
    # ============================================================================
    
    DATA_ROOT = get_data_root()
    IMAGES_DIR = DATA_ROOT / "raw" / "images"
    TAXONOMY_JSON = DATA_ROOT / "preprocessed" / "taxonomy_tree.json"
    LABELS_CSV = None  # Will be loaded from model_data_mapping.json
    
    # ============================================================================
    # Output Paths
    # ============================================================================
    
    OUTPUT_DIR = get_output_dir()
    CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
    EVAL_DIR = OUTPUT_DIR / "evaluation"
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    
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
    # Evaluation Parameters
    # ============================================================================
    
    # Beam Search (for hierarchical models)
    BEAM_WIDTH = 3
    
    # Verbose output
    VERBOSE = True
    
    @staticmethod
    def get_checkpoint_path(model_name):
        """Get checkpoint path for a model"""
        return EvaluationConfig.CHECKPOINT_PATHS.get(model_name)

