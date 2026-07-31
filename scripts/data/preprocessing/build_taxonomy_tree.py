#!/usr/bin/env python3
"""
Build Taxonomy Tree from Dataset

This script:
1. Extracts complete hierarchical relationships from labels_clean.csv
2. Builds a nested tree structure: Class -> Order -> Family -> Genus -> Species
3. Saves as JSON for reuse and validation
4. Generates statistics about the taxonomy

Usage:
    python -m scripts.data.preprocessing.build_taxonomy_tree
"""

import pandas as pd
import json
from pathlib import Path
from collections import defaultdict

# Paths
LABELS_CSV = Path("dataset/cleaned/labels_clean.csv")
OUTPUT_JSON = Path("dataset/preprocessed/taxonomy_tree.json")
OUTPUT_STATS = Path("dataset/preprocessed/taxonomy_stats.json")

def build_taxonomy_tree(df):
    """
    Build nested taxonomy tree from dataframe
    
    Structure:
    {
        "Class1": {
            "Order1": {
                "Family1": {
                    "Genus1": ["Species1", "Species2", ...],
                    "Genus2": ["Species3", ...]
                },
                "Family2": {...}
            },
            "Order2": {...}
        },
        ...
    }
    """
    tree = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(set))))
    
    for _, row in df.iterrows():
        class_name = row['class']
        order_name = row['order']
        family_name = row['family']
        genus_name = row['genus']
        species_name = row['species']
        
        # Add to tree (using species name)
        tree[class_name][order_name][family_name][genus_name].add(species_name)
    
    # Convert sets to sorted lists for JSON serialization
    result = {}
    for class_name, orders in sorted(tree.items()):
        result[class_name] = {}
        for order_name, families in sorted(orders.items()):
            result[class_name][order_name] = {}
            for family_name, genera in sorted(families.items()):
                result[class_name][order_name][family_name] = {}
                for genus_name, species_set in sorted(genera.items()):
                    result[class_name][order_name][family_name][genus_name] = sorted(species_set)
    
    return result

def build_flat_mappings(df):
    """
    Build flat mappings for quick lookup
    
    Returns:
    - class_to_orders: {class: [order1, order2, ...]}
    - order_to_families: {order: [family1, family2, ...]}
    - family_to_genera: {family: [genus1, genus2, ...]}
    - genus_to_species: {genus: [species1, species2, ...]}
    - class_to_all: {class: {order: {family: {genus: [species]}}}}
    """
    class_to_orders = defaultdict(set)
    order_to_families = defaultdict(set)
    family_to_genera = defaultdict(set)
    genus_to_species = defaultdict(set)
    
    for _, row in df.iterrows():
        class_to_orders[row['class']].add(row['order'])
        order_to_families[row['order']].add(row['family'])
        family_to_genera[row['family']].add(row['genus'])
        genus_to_species[row['genus']].add(row['species'])
    
    # Convert sets to sorted lists
    return {
        'class_to_orders': {k: sorted(v) for k, v in sorted(class_to_orders.items())},
        'order_to_families': {k: sorted(v) for k, v in sorted(order_to_families.items())},
        'family_to_genera': {k: sorted(v) for k, v in sorted(family_to_genera.items())},
        'genus_to_species': {k: sorted(v) for k, v in sorted(genus_to_species.items())},
    }

def calculate_statistics(df, tree):
    """Calculate taxonomy statistics"""
    stats = {
        'total_samples': len(df),
        'num_classes': df['class'].nunique(),
        'num_orders': df['order'].nunique(),
        'num_families': df['family'].nunique(),
        'num_genera': df['genus'].nunique(),
        'num_species': df['species'].nunique(),
        'class_distribution': df['class'].value_counts().to_dict(),
        'order_distribution': df['order'].value_counts().to_dict(),
        'family_distribution': df['family'].value_counts().to_dict(),
        'genus_distribution': df['genus'].value_counts().to_dict(),
        'species_distribution': df['species'].value_counts().to_dict(),
        'tree_structure': {
            class_name: {
                'num_orders': len(orders),
                'num_families': sum(len(families) for families in orders.values()),
                'num_genera': sum(
                    len(genera) 
                    for families in orders.values() 
                    for genera in families.values()
                ),
                'num_species': sum(
                    len(species) 
                    for families in orders.values() 
                    for genera in families.values() 
                    for species in genera.values()
                )
            }
            for class_name, orders in tree.items()
        }
    }
    
    # Convert numpy int64 to Python int for JSON serialization
    def convert_to_python_types(obj):
        if isinstance(obj, dict):
            return {k: convert_to_python_types(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert_to_python_types(item) for item in obj]
        elif hasattr(obj, 'item'):  # numpy types
            return obj.item()
        else:
            return obj
    
    return convert_to_python_types(stats)

def main():
    # Load data
    df = pd.read_csv(LABELS_CSV)
    
    # Build tree
    tree = build_taxonomy_tree(df)
    
    # Build flat mappings
    mappings = build_flat_mappings(df)
    
    # Calculate statistics
    stats = calculate_statistics(df, tree)
    
    # Combine everything
    output = {
        'tree': tree,
        'mappings': mappings,
        'statistics': stats,
        'metadata': {
            'source_file': str(LABELS_CSV),
            'description': 'Complete taxonomic hierarchy for DiatomScanNet dataset',
            'structure': 'Class -> Order -> Family -> Genus -> Species'
        }
    }
    
    # Save JSON
    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    # Save statistics separately
    OUTPUT_STATS.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_STATS, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    
    # Print summary
    print(f"Taxonomy tree built: {stats['total_samples']} samples | Classes: {stats['num_classes']} | "
          f"Orders: {stats['num_orders']} | Families: {stats['num_families']} | "
          f"Genera: {stats['num_genera']} | Species: {stats['num_species']}")

if __name__ == "__main__":
    main()

