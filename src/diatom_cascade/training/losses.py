"""
Focal Loss Functions for DiatomScanNet

All models use Focal Loss (α=0.25, γ=2.0) for fair comparison.
Used in training (for both training and validation loss calculation).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance
    
    FL(p_t) = -α * (1 - p_t)^γ * log(p_t)
    
    where:
        p_t = probability of true class
        α = balancing factor (default: 0.25)
        γ = focusing parameter (default: 2.0)
    """
    
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        """
        Args:
            inputs: (N, C) logits tensor
            targets: (N,) class indices tensor
        
        Returns:
            loss: scalar tensor
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)  # pt = probability of true class
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


def masked_focal_loss(logits, targets, mask, alpha=0.25, gamma=2.0):
    """
    Focal Loss with hierarchical masking for hierarchical models
    
    Applies hierarchical masking to restrict the candidate space,
    then computes Focal Loss.
    
    Args:
        logits: (batch_size, num_classes) - raw model outputs
        targets: (batch_size,) - ground truth class indices
        mask: (batch_size, num_classes) - binary mask (True=valid, False=invalid)
        alpha: focal loss alpha parameter (default: 0.25)
        gamma: focal loss gamma parameter (default: 2.0)
    
    Returns:
        loss: scalar tensor
    """
    # Apply mask: set invalid classes to very negative value
    very_neg = torch.finfo(logits.dtype).min / 2
    masked_logits = torch.where(mask, logits, very_neg)
    
    # Compute focal loss
    ce_loss = F.cross_entropy(masked_logits, targets, reduction='none')
    pt = torch.exp(-ce_loss)
    focal_loss = alpha * (1 - pt) ** gamma * ce_loss
    
    return focal_loss.mean()

