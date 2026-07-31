"""
Flat Bottom-up Lookup

Flat models predict only the finest level (species/genus), then use taxonomy tree
to look up upper levels. This is a bottom-up lookup approach.

Key characteristics:
- Model: Flat (single-level prediction)
- Lookup: Bottom-up (from fine to coarse using taxonomy tree)
- Uses hierarchy: Yes (during lookup, not during training)
- Method: Deterministic lookup (not model prediction)
"""

import json
from pathlib import Path


def load_taxonomy_tree(taxonomy_json_path):
    """
    Load taxonomy tree for bottom-up lookup
    
    Args:
        taxonomy_json_path: Path to taxonomy_tree.json
    
    Returns:
        tuple: (taxonomy_tree, mappings) or (None, None) if file not found
    """
    taxonomy_path = Path(taxonomy_json_path)
    if not taxonomy_path.exists():
        return None, None
    
    with open(taxonomy_path, 'r', encoding='utf-8') as f:
        taxonomy = json.load(f)
    return taxonomy.get('tree', {}), taxonomy.get('mappings', {})


def flat_bottom_up_lookup_from_species(species_name, taxonomy_tree, mappings):
    """
    Bottom-up lookup: Look up Class, Order, Family, Genus from Species
    using taxonomy tree (deterministic lookup, not model prediction)
    
    Args:
        species_name: Predicted species name
        taxonomy_tree: Taxonomy tree dictionary
        mappings: Taxonomy mappings dictionary
    
    Returns:
        tuple: (class_name, order_name, family_name, genus_name) or (None, None, None, None)
    """
    if taxonomy_tree is None:
        return None, None, None, None
    
    for class_name, orders in taxonomy_tree.items():
        for order_name, families in orders.items():
            for family_name, genera in families.items():
                for genus_name, species_list in genera.items():
                    if species_name in species_list:
                        return class_name, order_name, family_name, genus_name
    
    return None, None, None, None


def flat_bottom_up_lookup_from_genus(genus_name, taxonomy_tree, mappings):
    """
    Bottom-up lookup: Look up Class, Order, Family from Genus
    using taxonomy tree (deterministic lookup, not model prediction)
    
    Args:
        genus_name: Predicted genus name
        taxonomy_tree: Taxonomy tree dictionary
        mappings: Taxonomy mappings dictionary
    
    Returns:
        tuple: (class_name, order_name, family_name) or (None, None, None)
    """
    if taxonomy_tree is None:
        return None, None, None
    
    # Try tree traversal first
    for class_name, orders in taxonomy_tree.items():
        for order_name, families in orders.items():
            for family_name, genera in families.items():
                if genus_name in genera:
                    return class_name, order_name, family_name
    
    # Fallback to mappings if available
    if mappings and genus_name in mappings:
        mapping = mappings[genus_name]
        return mapping.get('class'), mapping.get('order'), mapping.get('family')
    
    return None, None, None


# Backward compatibility aliases
infer_upper_levels_from_species = flat_bottom_up_lookup_from_species
infer_upper_levels_from_genus = flat_bottom_up_lookup_from_genus
flat_bottom_up_infer_from_species = flat_bottom_up_lookup_from_species
flat_bottom_up_infer_from_genus = flat_bottom_up_lookup_from_genus

