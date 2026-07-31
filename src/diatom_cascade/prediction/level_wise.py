"""
Level-wise Argmax Prediction

每个层级独立进行argmax预测，不考虑层级约束。
这是最快的预测方法，但不保证层级一致性。

Method: Level-wise Independent Argmax

This method performs independent argmax at each level:
1. Predict class (argmax on class_logits)
2. Predict order (argmax on order_logits, independent of class)
3. Predict family (argmax on family_logits, independent of order)
4. Predict genus (argmax on genus_logits, independent of family)
5. Predict species (argmax on species_logits, independent of genus)

Key characteristics:
- Fastest method (O(n) complexity per level)
- No hierarchical constraints
- May produce biologically invalid combinations
- Useful as baseline comparison
"""
import torch


def level_wise_argmax_predict(class_logits, order_logits, family_logits=None, 
                               genus_logits=None, species_logits=None):
    """
    Level-wise Argmax Prediction for multi-level classification
    
    Each level independently selects the class with highest probability,
    without considering hierarchical constraints.
    
    Args:
        class_logits: [B, num_classes] - Class logits
        order_logits: [B, num_orders] - Order logits
        family_logits: [B, num_families] - Family logits (optional, for 3+ levels)
        genus_logits: [B, num_genera] - Genus logits (optional, for 4+ levels)
        species_logits: [B, num_species] - Species logits (optional, for 5 levels)
    
    Returns:
        Tuple of predicted indices for each level
    
    Examples:
        # 2 levels: Class -> Order
        pred_class, pred_order = level_wise_argmax_predict(
            class_logits, order_logits
        )
        
        # 3 levels: Class -> Order -> Family
        pred_class, pred_order, pred_family = level_wise_argmax_predict(
            class_logits, order_logits, family_logits
        )
        
        # 4 levels: Class -> Order -> Family -> Genus
        pred_class, pred_order, pred_family, pred_genus = level_wise_argmax_predict(
            class_logits, order_logits, family_logits, genus_logits
        )
        
        # 5 levels: Class -> Order -> Family -> Genus -> Species
        pred_class, pred_order, pred_family, pred_genus, pred_species = level_wise_argmax_predict(
            class_logits, order_logits, family_logits, genus_logits, species_logits
        )
    """
    # Determine number of levels
    logits_list = [class_logits, order_logits]
    
    if family_logits is not None:
        logits_list.append(family_logits)
    
    if genus_logits is not None:
        logits_list.append(genus_logits)
    
    if species_logits is not None:
        logits_list.append(species_logits)
    
    num_levels = len(logits_list)
    B = class_logits.shape[0]
    device = class_logits.device
    
    # Initialize result tensors
    results = [torch.empty(B, dtype=torch.long, device=device) for _ in range(num_levels)]
    
    # Level-wise independent argmax at each level
    for level in range(num_levels):
        results[level] = torch.argmax(logits_list[level], dim=1)
    
    return tuple(results)


# Backward compatibility alias (deprecated, use level_wise_argmax_predict instead)
argmax_independent_predict = level_wise_argmax_predict

