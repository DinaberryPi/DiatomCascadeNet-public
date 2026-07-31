"""
Beam Search Constrained Hierarchical Prediction

使用束搜索找到全局最优的层级预测路径
"""

import torch
import torch.nn.functional as F


def beam_search_hierarchical_predict(class_logits, order_logits, family_logits=None, 
                                     genus_logits=None, species_logits=None,
                                     M_class_order=None, M_order_family=None, 
                                     M_family_genus=None, M_genus_species=None,
                                     beam_width=3):
    """
    Beam Search预测：联合优化所有层级，找到全局最优路径
    
    Args:
        class_logits: (B, num_classes) - Class层级的logits
        order_logits: (B, num_orders) - Order层级的logits
        family_logits: (B, num_families) - Family层级的logits (可选)
        genus_logits: (B, num_genera) - Genus层级的logits (可选)
        species_logits: (B, num_species) - Species层级的logits (可选)
        M_class_order: (num_classes, num_orders) - Class到Order的mask矩阵
        M_order_family: (num_orders, num_families) - Order到Family的mask矩阵
        M_family_genus: (num_families, num_genera) - Family到Genus的mask矩阵
        M_genus_species: (num_genera, num_species) - Genus到Species的mask矩阵
        beam_width: Beam search的宽度（默认3）
    
    Returns:
        tuple: (pred_class, pred_order, pred_family, pred_genus, pred_species)
               每个都是(B,)的tensor，包含预测的类别索引
    """
    device = class_logits.device
    B = class_logits.shape[0]
    
    # 转换为log probabilities
    class_logprobs = F.log_softmax(class_logits, dim=1)  # (B, num_classes)
    order_logprobs = F.log_softmax(order_logits, dim=1)   # (B, num_orders)
    
    # 确定层级数量
    num_levels = 2  # Class, Order
    logprobs_list = [class_logprobs, order_logprobs]
    masks_list = [M_class_order]
    
    if family_logits is not None:
        num_levels = 3
        family_logprobs = F.log_softmax(family_logits, dim=1)
        logprobs_list.append(family_logprobs)
        masks_list.append(M_order_family)
    
    if genus_logits is not None:
        num_levels = 4
        genus_logprobs = F.log_softmax(genus_logits, dim=1)
        logprobs_list.append(genus_logprobs)
        masks_list.append(M_family_genus)
    
    if species_logits is not None:
        num_levels = 5
        species_logprobs = F.log_softmax(species_logits, dim=1)
        logprobs_list.append(species_logprobs)
        masks_list.append(M_genus_species)
    
    # 初始化结果tensor
    results = []
    for i in range(num_levels):
        results.append(torch.zeros(B, dtype=torch.long, device=device))
    
    very_neg = torch.finfo(class_logprobs.dtype).min / 2
    
    # 对每个样本进行beam search
    for b in range(B):
        # Level 1: Class (无约束)
        # 选择top-k classes作为初始beam
        top_k_class_probs, top_k_classes = torch.topk(class_logprobs[b], k=min(beam_width, class_logprobs.shape[1]))
        
        # Beam: [(class_idx, order_idx, family_idx, genus_idx, species_idx, log_prob)]
        # 根据层级数量初始化路径
        path_template = [None] * num_levels + [0.0]  # 最后一个是log_prob
        beam = []
        for c, p in zip(top_k_classes, top_k_class_probs):
            c_val = int(c.item())
            # Ensure class index is valid (non-negative)
            if c_val < 0 or c_val >= class_logprobs.shape[1]:
                continue  # Skip invalid indices
            path = path_template.copy()
            path[0] = c_val
            path[-1] = float(p.item())
            beam.append(tuple(path))
        
        # 逐层扩展beam
        for level in range(1, num_levels):
            new_beam = []
            current_logprobs = logprobs_list[level][b]  # (num_classes_at_level,)
            current_mask = masks_list[level - 1]  # (num_prev_classes, num_current_classes)
            
            for path in beam:
                prev_idx = path[level - 1]  # 上一层的索引
                
                # 获取mask
                if prev_idx is not None:
                    # Ensure prev_idx is valid (non-negative and within bounds)
                    # Convert to int if it's a tensor or other type
                    if isinstance(prev_idx, torch.Tensor):
                        prev_idx = int(prev_idx.item())
                    elif not isinstance(prev_idx, int):
                        try:
                            prev_idx = int(prev_idx)
                        except (ValueError, TypeError):
                            prev_idx = None
                    
                    if prev_idx is not None and isinstance(prev_idx, int) and prev_idx >= 0 and prev_idx < current_mask.shape[0]:
                        mask = current_mask[prev_idx]  # (num_current_classes,)
                        # 应用mask
                        masked_logprobs = torch.where(mask, current_logprobs, very_neg)
                    else:
                        # Invalid index, use all logprobs (no masking)
                        masked_logprobs = current_logprobs
                else:
                    masked_logprobs = current_logprobs
                
                # 选择top-k候选
                top_k = min(beam_width, masked_logprobs.shape[0])
                top_k_probs, top_k_indices = torch.topk(masked_logprobs, k=top_k)
                
                # 扩展路径
                for prob, idx in zip(top_k_probs, top_k_indices):
                    idx_val = int(idx.item())
                    # Ensure index is valid (non-negative and within bounds)
                    if idx_val < 0 or idx_val >= current_logprobs.shape[0]:
                        continue  # Skip invalid indices
                    new_path = list(path)
                    new_path[level] = idx_val
                    new_path[-1] += float(prob.item())  # 累加log概率
                    new_beam.append(tuple(new_path))
            
            # 保留top-k路径
            new_beam.sort(key=lambda x: x[-1], reverse=True)
            beam = new_beam[:beam_width]
        
        # 选择最优路径
        if len(beam) == 0:
            # Fallback: use argmax if beam is empty
            for level in range(num_levels):
                results[level][b] = torch.argmax(logprobs_list[level][b]).item()
        else:
            best_path = beam[0]
            for level in range(num_levels):
                idx_val = best_path[level]
                # Ensure index is valid
                if idx_val is not None:
                    # Convert to int if it's a tensor or other type
                    if isinstance(idx_val, torch.Tensor):
                        idx_val = int(idx_val.item())
                    elif not isinstance(idx_val, int):
                        try:
                            idx_val = int(idx_val)
                        except (ValueError, TypeError):
                            idx_val = None
                    
                    if idx_val is not None and isinstance(idx_val, int) and idx_val >= 0 and idx_val < logprobs_list[level].shape[1]:
                        results[level][b] = idx_val
                    else:
                        # Fallback to argmax if invalid
                        results[level][b] = torch.argmax(logprobs_list[level][b]).item()
                else:
                    # Fallback to argmax if None
                    results[level][b] = torch.argmax(logprobs_list[level][b]).item()
    
    return tuple(results)


def beam_search_hierarchical_predict_4level(class_logits, order_logits, family_logits, genus_logits,
                                            M_class_order, M_order_family, M_family_genus,
                                            beam_width=3):
    """
    4层级Beam Search层级预测的便捷函数（Class → Order → Family → Genus）
    """
    return beam_search_hierarchical_predict(
        class_logits, order_logits,
        family_logits=family_logits, genus_logits=genus_logits,
        M_class_order=M_class_order, M_order_family=M_order_family,
        M_family_genus=M_family_genus,
        beam_width=beam_width
    )


# Backward compatibility alias (deprecated, use beam_search_hierarchical_predict_4level instead)
beam_search_predict_4level = beam_search_hierarchical_predict_4level

