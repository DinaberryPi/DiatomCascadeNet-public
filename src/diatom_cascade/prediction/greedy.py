"""
Greedy Hierarchical Constrained Prediction

支持 2-5 层级的层级约束贪婪预测
使用层级mask约束，逐层贪婪选择最优选项

Method: Hierarchical Constrained Greedy Decoding

This method performs greedy top-down prediction with hierarchical constraints:
1. Predict class (argmax on class_logits)
2. Predict order (argmax on order_logits, masked by predicted class)
3. Predict family (argmax on family_logits, masked by predicted order)
4. Predict genus (argmax on genus_logits, masked by predicted family)
5. Predict species (argmax on species_logits, masked by predicted genus)

Key advantages:
- Fast and simple (O(n) complexity per level)
- Guarantees biological consistency (mask constraints)
- No beam search overhead

Note: This is a greedy method that selects the best option at each level
with mask constraints from upper levels, which may not find the globally optimal path.
"""
import torch


def greedy_hierarchical_predict(class_logits, order_logits, family_logits=None, 
                                 genus_logits=None, species_logits=None,
                                 M_class_order=None, M_order_family=None, 
                                 M_family_genus=None, M_genus_species=None):
    """
    Greedy Hierarchical Constrained Prediction for multi-level classification
    
    This method performs greedy top-down prediction with hierarchical constraints:
    1. Predict class (argmax on class_logits)
    2. Predict order (argmax on order_logits, masked by predicted class)
    3. Predict family (argmax on family_logits, masked by predicted order)
    4. Predict genus (argmax on genus_logits, masked by predicted family)
    5. Predict species (argmax on species_logits, masked by predicted genus)
    
    Args:
        class_logits: [B, num_classes] - Class logits
        order_logits: [B, num_orders] - Order logits
        family_logits: [B, num_families] - Family logits (optional, for 3+ levels)
        genus_logits: [B, num_genera] - Genus logits (optional, for 4+ levels)
        species_logits: [B, num_species] - Species logits (optional, for 5 levels)
        M_class_order: [num_classes, num_orders] - Mask matrix
        M_order_family: [num_orders, num_families] - Mask matrix (optional)
        M_family_genus: [num_families, num_genera] - Mask matrix (optional)
        M_genus_species: [num_genera, num_species] - Mask matrix (optional)
    
    Returns:
        Tuple of predicted indices for each level
    
    Examples:
        # 2 levels: Class -> Order
        pred_class, pred_order = greedy_hierarchical_predict(
            class_logits, order_logits,
            M_class_order=M_class_order
        )
        
        # 3 levels: Class -> Order -> Family
        pred_class, pred_order, pred_family = greedy_hierarchical_predict(
            class_logits, order_logits, family_logits,
            M_class_order=M_class_order, M_order_family=M_order_family
        )
        
        # 4 levels: Class -> Order -> Family -> Genus
        pred_class, pred_order, pred_family, pred_genus = greedy_hierarchical_predict(
            class_logits, order_logits, family_logits, genus_logits,
            M_class_order=M_class_order, M_order_family=M_order_family, 
            M_family_genus=M_family_genus
        )
        
        # 5 levels: Class -> Order -> Family -> Genus -> Species
        pred_class, pred_order, pred_family, pred_genus, pred_species = greedy_hierarchical_predict(
            class_logits, order_logits, family_logits, genus_logits, species_logits,
            M_class_order=M_class_order, M_order_family=M_order_family, 
            M_family_genus=M_family_genus, M_genus_species=M_genus_species
        )
    """
    # Determine number of levels
    logits_list = [class_logits, order_logits]
    masks_list = []
    
    if family_logits is not None:
        logits_list.append(family_logits)
        if M_order_family is None:
            raise ValueError("M_order_family must be provided when family_logits is provided")
        masks_list.append(M_order_family)
    
    if genus_logits is not None:
        logits_list.append(genus_logits)
        if M_family_genus is None:
            raise ValueError("M_family_genus must be provided when genus_logits is provided")
        masks_list.append(M_family_genus)
    
    if species_logits is not None:
        logits_list.append(species_logits)
        if M_genus_species is None:
            raise ValueError("M_genus_species must be provided when species_logits is provided")
        masks_list.append(M_genus_species)
    
    if M_class_order is None:
        raise ValueError("M_class_order must be provided")
    masks_list.insert(0, M_class_order)
    
    num_levels = len(logits_list)
    B = class_logits.shape[0]
    device = class_logits.device
    
    # Convert all logits to log probabilities
    logps = [torch.log_softmax(logits, dim=1) for logits in logits_list]
    
    # Convert masks to device
    masks = [M.to(device) for M in masks_list]
    
    # Initialize result tensors
    results = [torch.empty(B, dtype=torch.long, device=device) for _ in range(num_levels)]
    
    very_neg = torch.finfo(logps[0].dtype).min / 2
    
    # Greedy top-down prediction
    for b in range(B):
        # Level 1: Class (no constraint)
        c = torch.argmax(logps[0][b]).item()
        results[0][b] = c
        
        # Level 2: Order (constrained by class)
        mask_o = masks[0][c]
        masked_o = torch.where(mask_o, logps[1][b], very_neg)
        o = torch.argmax(masked_o).item()
        results[1][b] = o
        
        if num_levels >= 3:
            # Level 3: Family (constrained by order)
            mask_f = masks[1][o]
            masked_f = torch.where(mask_f, logps[2][b], very_neg)
            f = torch.argmax(masked_f).item()
            results[2][b] = f
            
            if num_levels >= 4:
                # Level 4: Genus (constrained by family)
                mask_g = masks[2][f]
                masked_g = torch.where(mask_g, logps[3][b], very_neg)
                g = torch.argmax(masked_g).item()
                results[3][b] = g
                
                if num_levels >= 5:
                    # Level 5: Species (constrained by genus)
                    mask_s = masks[3][g]
                    masked_s = torch.where(mask_s, logps[4][b], very_neg)
                    s = torch.argmax(masked_s).item()
                    results[4][b] = s
    
    return tuple(results)


# Backward compatibility alias (deprecated, use greedy_hierarchical_predict instead)
hierarchical_predict = greedy_hierarchical_predict

