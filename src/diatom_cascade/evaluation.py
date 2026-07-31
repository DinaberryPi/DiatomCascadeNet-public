"""
Evaluation Utilities

Functions specific to the evaluation phase.
"""

from torch.utils.data import DataLoader
from .config.evaluation_config import EvaluationConfig


def create_test_loader(dataset, batch_size=None, num_workers=None):
    """
    Create test DataLoader with standardized settings
    
    Args:
        dataset: PyTorch Dataset
        batch_size: Batch size (defaults to EvaluationConfig.BATCH_SIZE)
        num_workers: Number of workers (defaults to EvaluationConfig.NUM_WORKERS)
    
    Returns:
        DataLoader
    """
    from torch.utils.data import DataLoader
    
    if batch_size is None:
        batch_size = EvaluationConfig.BATCH_SIZE
    if num_workers is None:
        num_workers = EvaluationConfig.NUM_WORKERS
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=EvaluationConfig.PIN_MEMORY
    )

