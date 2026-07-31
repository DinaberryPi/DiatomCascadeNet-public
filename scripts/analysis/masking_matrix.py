#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualize Masking Matrix as Relationship Tree

This script visualizes how masking matrices represent the hierarchical
relationship tree in matrix form.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import sys
import io

# Set UTF-8 encoding for output
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# Load taxonomy
TAXONOMY_JSON = Path("dataset/taxonomy_tree.json")
LABELS_CSV = Path("dataset/labels_clean.csv")

def build_masking_matrices():
    """Build masking matrices from data"""
    df = pd.read_csv(LABELS_CSV)
    
    class_encoder = LabelEncoder()
    class_encoder.fit(df['class'].unique())
    
    order_encoder = LabelEncoder()
    order_encoder.fit(df['order'].unique())
    
    family_encoder = LabelEncoder()
    family_encoder.fit(df['family'].unique())
    
    # Build ID mappings
    class2id = {c: i for i, c in enumerate(class_encoder.classes_)}
    order2id = {o: i for i, o in enumerate(order_encoder.classes_)}
    family2id = {f: i for i, f in enumerate(family_encoder.classes_)}
    
    # Build matrices
    M_class_order = np.zeros((len(class_encoder.classes_), len(order_encoder.classes_)), dtype=bool)
    M_order_family = np.zeros((len(order_encoder.classes_), len(family_encoder.classes_)), dtype=bool)
    
    for _, row in df.iterrows():
        c = class2id[row['class']]
        o = order2id[row['order']]
        f = family2id[row['family']]
        
        M_class_order[c, o] = True
        M_order_family[o, f] = True
    
    return {
        'M_class_order': M_class_order,
        'M_order_family': M_order_family,
        'class_names': class_encoder.classes_,
        'order_names': order_encoder.classes_,
        'family_names': family_encoder.classes_
    }

def visualize_masking_matrix_as_tree():
    """Visualize masking matrix showing the relationship tree"""
    print("=" * 70)
    print("  Visualizing Masking Matrix as Relationship Tree")
    print("=" * 70)
    
    # Load taxonomy tree
    with open(TAXONOMY_JSON, 'r', encoding='utf-8') as f:
        taxonomy = json.load(f)
    
    # Build masking matrices
    masks = build_masking_matrices()
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 12))
    
    # Plot 1: M_class_order heatmap
    ax1 = plt.subplot(2, 2, 1)
    sns.heatmap(
        masks['M_class_order'].astype(int),
        annot=True,
        fmt='d',
        cmap='YlOrRd',
        xticklabels=masks['order_names'],
        yticklabels=masks['class_names'],
        cbar_kws={'label': 'Valid Relationship (1=Yes, 0=No)'},
        ax=ax1
    )
    ax1.set_title('M_class_order: Class → Order Relationship Matrix\n(Each row is a Class, each column is an Order)', 
                  fontsize=12, fontweight='bold')
    ax1.set_xlabel('Orders', fontsize=10)
    ax1.set_ylabel('Classes', fontsize=10)
    plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')
    plt.setp(ax1.get_yticklabels(), rotation=0)
    
    # Plot 2: M_order_family heatmap
    ax2 = plt.subplot(2, 2, 2)
    sns.heatmap(
        masks['M_order_family'].astype(int),
        annot=True,
        fmt='d',
        cmap='YlGnBu',
        xticklabels=masks['family_names'],
        yticklabels=masks['order_names'],
        cbar_kws={'label': 'Valid Relationship (1=Yes, 0=No)'},
        ax=ax2
    )
    ax2.set_title('M_order_family: Order → Family Relationship Matrix\n(Each row is an Order, each column is a Family)', 
                  fontsize=12, fontweight='bold')
    ax2.set_xlabel('Families', fontsize=10)
    ax2.set_ylabel('Orders', fontsize=10)
    plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')
    plt.setp(ax2.get_yticklabels(), rotation=0)
    
    # Plot 3: Tree structure visualization (text)
    ax3 = plt.subplot(2, 2, 3)
    ax3.axis('off')
    
    tree_text = "Relationship Tree Structure:\n\n"
    tree = taxonomy['tree']
    for class_name, orders in list(tree.items())[:3]:  # Show first 3 classes
        tree_text += f"[Class] {class_name}\n"
        for order_name, families in list(orders.items())[:2]:  # Show first 2 orders per class
            tree_text += f"  +-- [Order] {order_name}\n"
            for family_name, genera in list(families.items())[:2]:  # Show first 2 families per order
                tree_text += f"     +-- [Family] {family_name}\n"
                for genus_name, species_list in list(genera.items())[:1]:  # Show first 1 genus per family
                    tree_text += f"        +-- [Genus] {genus_name} ({len(species_list)} species)\n"
        tree_text += "\n"
    
    tree_text += "... (truncated for display)\n\n"
    tree_text += "Note: This tree structure is represented as masking matrices above!"
    
    ax3.text(0.05, 0.95, tree_text, transform=ax3.transAxes, 
             fontsize=10, verticalalignment='top', family='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax3.set_title('Tree Structure (Nested)', fontsize=12, fontweight='bold')
    
    # Plot 4: Explanation
    ax4 = plt.subplot(2, 2, 4)
    ax4.axis('off')
    
    explanation = """
    Masking Matrix = Relationship Tree in Matrix Form
    
    Key Points:
    
    1. Tree Structure (Nested):
       Class -> Order -> Family -> Genus -> Species
       (Hierarchical, nested dictionary)
    
    2. Masking Matrix (Flat):
       M[i, j] = True if relationship exists
       (2D boolean matrix, easy for computation)
    
    3. Equivalence:
       - Tree: "Bacillariophyceae contains Naviculales"
       - Matrix: M_class_order[0, 5] = True
         (if Bacillariophyceae is class 0, 
          Naviculales is order 5)
    
    4. Advantages of Matrix Form:
       + Fast matrix operations (GPU-friendly)
       + Easy masking in neural networks
       + Efficient batch processing
       + Direct indexing for lookup
    
    5. Usage in Training:
       - Mask invalid predictions
       - Enforce hierarchical constraints
       - Guide model learning
    """
    
    ax4.text(0.05, 0.95, explanation, transform=ax4.transAxes,
             fontsize=10, verticalalignment='top', family='monospace',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    ax4.set_title('Matrix vs Tree Representation', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    
    # Save
    output_path = Path("report/masking_matrix_as_tree.png")
    output_path.parent.mkdir(exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n[OK] Saved visualization to {output_path}")
    
    # Also create a detailed example
    create_detailed_example(masks, taxonomy)
    
    return fig

def create_detailed_example(masks, taxonomy):
    """Create a detailed example showing one path through the tree"""
    print("\n" + "=" * 70)
    print("  Detailed Example: One Path Through the Tree")
    print("=" * 70)
    
    # Pick an example: Bacillariophyceae -> Naviculales -> Naviculaceae
    class_name = "Bacillariophyceae"
    order_name = "Naviculales"
    family_name = "Naviculaceae"
    
    class_idx = list(masks['class_names']).index(class_name)
    order_idx = list(masks['order_names']).index(order_name)
    family_idx = list(masks['family_names']).index(family_name)
    
    print(f"\nExample Path:")
    print(f"  {class_name} -> {order_name} -> {family_name}")
    print(f"\nIn Masking Matrices:")
    print(f"  M_class_order[{class_idx}, {order_idx}] = {masks['M_class_order'][class_idx, order_idx]}")
    print(f"  M_order_family[{order_idx}, {family_idx}] = {masks['M_order_family'][order_idx, family_idx]}")
    
    print(f"\nIn Tree Structure:")
    tree = taxonomy['tree']
    if class_name in tree and order_name in tree[class_name] and family_name in tree[class_name][order_name]:
        genera = tree[class_name][order_name][family_name]
        print(f"  {class_name}")
        print(f"    +-- {order_name}")
        print(f"       +-- {family_name}")
        print(f"          +-- {len(genera)} genera:")
        for genus_name, species_list in list(genera.items())[:3]:
            try:
                print(f"             - {genus_name} ({len(species_list)} species)")
            except UnicodeEncodeError:
                print(f"             - {genus_name.encode('ascii', 'replace').decode('ascii')} ({len(species_list)} species)")
        if len(genera) > 3:
            print(f"             ... and {len(genera) - 3} more genera")
    
    print(f"\nNote: The masking matrix encodes this entire tree structure!")
    print(f"      Each 'True' value represents a valid parent-child relationship.")

def main():
    fig = visualize_masking_matrix_as_tree()
    print("\n" + "=" * 70)
    print("  Visualization Complete!")
    print("=" * 70)
    print(f"\nThe masking matrix IS the relationship tree in matrix form!")
    print(f"Open report/masking_matrix_as_tree.png to see the visualization.")
    plt.close()

if __name__ == "__main__":
    main()

