#!/usr/bin/env python3
"""
Load Hierarchy Masks Utility

This utility module provides functions to build hierarchical constraint matrices
for training and evaluation scripts. This eliminates code duplication across
all training scripts.

Two functions are provided:
1. build_hierarchy_masks_from_dataframe() - For training scripts (uses filtered data)
2. load_hierarchy_masks() - For other use cases (loads from taxonomy_tree.json)

Usage in training scripts:
    from utils.common import build_hierarchy_masks_from_dataframe
    
    # After filtering data and creating encoders
    masks = build_hierarchy_masks_from_dataframe(
        df, class_encoder, order_encoder, family_encoder, genus_encoder
    )
    Config.M_CLASS_ORDER = masks['M_class_order']
    Config.M_ORDER_FAMILY = masks['M_order_family']
    Config.M_FAMILY_GENUS = masks['M_family_genus']
"""

import json
import numpy as np
import torch
from pathlib import Path
from typing import Dict, Any, Optional
from ..config.path_config import get_data_root

# Path to taxonomy tree
TAXONOMY_JSON = get_data_root() / "preprocessed" / "taxonomy_tree.json"

def load_hierarchy_masks(
    class_encoder,
    order_encoder,
    family_encoder,
    genus_encoder: Optional[Any] = None,
    species_encoder: Optional[Any] = None
) -> Dict[str, torch.Tensor]:
    """
    Load hierarchical constraint matrices from taxonomy_tree.json
    
    Args:
        class_encoder: LabelEncoder for classes
        order_encoder: LabelEncoder for orders
        family_encoder: LabelEncoder for families
        genus_encoder: Optional LabelEncoder for genera
        species_encoder: Optional LabelEncoder for species
        
    Returns:
        Dictionary containing constraint matrices as torch.Tensor:
        - M_class_order: [num_classes, num_orders] bool
        - M_order_family: [num_orders, num_families] bool
        - M_family_genus: [num_families, num_genera] bool (if genus_encoder provided)
        - M_genus_species: [num_genera, num_species] bool (if species_encoder provided)
    """
    # Load taxonomy tree
    if not TAXONOMY_JSON.exists():
        raise FileNotFoundError(
            f"Taxonomy tree not found at {TAXONOMY_JSON}. "
            f"Please run: python -m scripts.data.preprocessing.build_taxonomy_tree"
        )
    
    with open(TAXONOMY_JSON, 'r', encoding='utf-8') as f:
        taxonomy = json.load(f)
    
    tree = taxonomy['tree']
    mappings = taxonomy['mappings']
    
    # Build ID mappings
    class2id = {c: i for i, c in enumerate(class_encoder.classes_)}
    order2id = {o: i for i, o in enumerate(order_encoder.classes_)}
    family2id = {f: i for i, f in enumerate(family_encoder.classes_)}
    
    num_classes = len(class_encoder.classes_)
    num_orders = len(order_encoder.classes_)
    num_families = len(family_encoder.classes_)
    
    # Initialize matrices
    masks = {}
    
    # Build M_class_order: [num_classes, num_orders]
    M_class_order = np.zeros((num_classes, num_orders), dtype=bool)
    for class_name, orders in tree.items():
        if class_name in class2id:
            class_idx = class2id[class_name]
            for order_name in orders.keys():
                if order_name in order2id:
                    order_idx = order2id[order_name]
                    M_class_order[class_idx, order_idx] = True
    masks['M_class_order'] = torch.from_numpy(M_class_order)
    
    # Build M_order_family: [num_orders, num_families]
    M_order_family = np.zeros((num_orders, num_families), dtype=bool)
    for order_name, families in mappings['order_to_families'].items():
        if order_name in order2id:
            order_idx = order2id[order_name]
            for family_name in families:
                if family_name in family2id:
                    family_idx = family2id[family_name]
                    M_order_family[order_idx, family_idx] = True
    masks['M_order_family'] = torch.from_numpy(M_order_family)
    
    # Build M_family_genus: [num_families, num_genera] (if genus_encoder provided)
    if genus_encoder is not None:
        genus2id = {g: i for i, g in enumerate(genus_encoder.classes_)}
        num_genera = len(genus_encoder.classes_)
        M_family_genus = np.zeros((num_families, num_genera), dtype=bool)
        
        for family_name, genera in mappings['family_to_genera'].items():
            if family_name in family2id:
                family_idx = family2id[family_name]
                for genus_name in genera:
                    if genus_name in genus2id:
                        genus_idx = genus2id[genus_name]
                        M_family_genus[family_idx, genus_idx] = True
        masks['M_family_genus'] = torch.from_numpy(M_family_genus)
    
    # Build M_genus_species: [num_genera, num_species] (if species_encoder provided)
    if species_encoder is not None and genus_encoder is not None:
        genus2id = {g: i for i, g in enumerate(genus_encoder.classes_)}
        species2id = {s: i for i, s in enumerate(species_encoder.classes_)}
        num_genera = len(genus_encoder.classes_)
        num_species = len(species_encoder.classes_)
        M_genus_species = np.zeros((num_genera, num_species), dtype=bool)
        
        # Build from tree structure
        for class_name, orders in tree.items():
            for order_name, families in orders.items():
                for family_name, genera in families.items():
                    for genus_name, species_list in genera.items():
                        if genus_name in genus2id:
                            genus_idx = genus2id[genus_name]
                            for species_name in species_list:
                                if species_name in species2id:
                                    species_idx = species2id[species_name]
                                    M_genus_species[genus_idx, species_idx] = True
        masks['M_genus_species'] = torch.from_numpy(M_genus_species)
    
    return masks

def build_hierarchy_masks_from_dataframe(
    df,
    class_encoder,
    order_encoder,
    family_encoder: Optional[Any] = None,
    genus_encoder: Optional[Any] = None,
    species_encoder: Optional[Any] = None
) -> Dict[str, torch.Tensor]:
    """
    Build hierarchical constraint matrices directly from dataframe.
    This is the original method used in training scripts.
    
    This function is kept for backward compatibility, but it's recommended
    to use load_hierarchy_masks() instead for consistency.
    """
    masks = {}
    
    # Build ID mappings
    class2id = {c: i for i, c in enumerate(class_encoder.classes_)}
    order2id = {o: i for i, o in enumerate(order_encoder.classes_)}
    
    num_classes = len(class_encoder.classes_)
    num_orders = len(order_encoder.classes_)
    
    # Build M_class_order
    M_class_order = np.zeros((num_classes, num_orders), dtype=bool)
    for _, row in df.iterrows():
        c = class2id[row['class']]
        o = order2id.get(row['order'])
        if o is not None:
            M_class_order[c, o] = True
    masks['M_class_order'] = torch.from_numpy(M_class_order)
    
    # Build M_order_family (if family_encoder provided)
    if family_encoder is not None:
        family2id = {f: i for i, f in enumerate(family_encoder.classes_)}
        num_families = len(family_encoder.classes_)
        M_order_family = np.zeros((num_orders, num_families), dtype=bool)
        for _, row in df.iterrows():
            o = order2id.get(row['order'])
            f = family2id.get(row['family'])
            if o is not None and f is not None:
                M_order_family[o, f] = True
        masks['M_order_family'] = torch.from_numpy(M_order_family)
    
    # Build M_family_genus (if provided)
    if genus_encoder is not None and family_encoder is not None:
        genus2id = {g: i for i, g in enumerate(genus_encoder.classes_)}
        num_genera = len(genus_encoder.classes_)
        num_families = len(family_encoder.classes_)
        M_family_genus = np.zeros((num_families, num_genera), dtype=bool)
        for _, row in df.iterrows():
            f = family2id.get(row['family'])
            g = genus2id.get(row['genus'])
            if f is not None and g is not None:
                M_family_genus[f, g] = True
        masks['M_family_genus'] = torch.from_numpy(M_family_genus)
    
    # Build M_genus_species (if provided)
    if species_encoder is not None and genus_encoder is not None:
        genus2id = {g: i for i, g in enumerate(genus_encoder.classes_)}
        species2id = {s: i for i, s in enumerate(species_encoder.classes_)}
        num_genera = len(genus_encoder.classes_)
        num_species = len(species_encoder.classes_)
        M_genus_species = np.zeros((num_genera, num_species), dtype=bool)
        for _, row in df.iterrows():
            g = genus2id[row['genus']]
            s = species2id[row['species']]
            M_genus_species[g, s] = True
        masks['M_genus_species'] = torch.from_numpy(M_genus_species)
    
    return masks

