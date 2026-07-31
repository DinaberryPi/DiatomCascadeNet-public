#!/usr/bin/env python3
"""
Create Filtered Datasets for Hierarchical Models

This script creates progressive filtered datasets for different model types:
- labels_clean_CO.csv: For H-CO (filter Order >= MIN_SAMPLES["Order"])
- labels_clean_COF.csv: For H-COF (filter Order >= MIN_SAMPLES["Order"], Family >= MIN_SAMPLES["Family"])
- labels_clean_COFG.csv: For H-COFG (filter Order >= MIN_SAMPLES["Order"], Family >= MIN_SAMPLES["Family"], Genus >= MIN_SAMPLES["Genus"])
- labels_clean_COFGS.csv: For H-COFGS and F-S (filter all levels >= MIN_SAMPLES thresholds)

Key Design:
- Progressive filtering: Each dataset builds on the previous one
- MIN_SAMPLES thresholds are read from the dependency-free data config
- Critical pair (H-COFGS vs F-S) uses identical files

Usage:
    python -m scripts.data.preprocessing.create_filtered_datasets
"""

import pandas as pd
from pathlib import Path
import json
import sys
import io

# Fix encoding for Windows console
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from diatom_cascade.config.data_config import MIN_SAMPLES

# Paths
DATA_ROOT = Path("dataset")
LABELS_CLEAN_CSV = DATA_ROOT / "cleaned" / "labels_clean.csv"
OUTPUT_DIR = DATA_ROOT / "preprocessed"

def remove_uncertain_species(df):
    """
    Remove uncertain species annotations (sp., cf., aff., etc.)
    These are not valid taxonomic names and should be filtered out.
    """
    if 'species' not in df.columns:
        return df
    
    # Patterns for uncertain species
    uncertain_patterns = [
        'sp.', 'sp', 'cf.', 'cf', 'aff.', 'aff', 
        'sus.', 'sus', '?', 'unknown', 'indet.'
    ]
    
    initial_count = len(df)
    
    # Filter out rows where species contains uncertain patterns
    mask = df['species'].notna() & (df['species'] != '')
    for pattern in uncertain_patterns:
        mask = mask & ~df['species'].str.contains(pattern, case=False, na=False, regex=False)
    
    df_filtered = df[mask].copy()
    removed_count = initial_count - len(df_filtered)
    
    if removed_count > 0:
        print(f"Removed {removed_count} samples with uncertain species annotations")
    
    return df_filtered


def create_co_dataset(df_base):
    """Create dataset for H-CO (Class → Order)"""
    df = df_base.copy()
    order_counts = df['order'].value_counts()
    valid_orders = order_counts[order_counts >= MIN_SAMPLES["Order"]].index
    df = df[df['order'].isin(valid_orders)].copy()
    return df


def create_cof_dataset(df_co):
    """Create dataset for H-COF (Class → Order → Family)"""
    df = df_co.copy()
    family_counts = df['family'].value_counts()
    valid_families = family_counts[family_counts >= MIN_SAMPLES["Family"]].index
    df = df[df['family'].isin(valid_families)].copy()
    return df


def create_cofg_dataset(df_cof):
    """Create dataset for H-COFG (Genus level)"""
    df = df_cof.copy()
    genus_counts = df['genus'].value_counts()
    valid_genera = genus_counts[genus_counts >= MIN_SAMPLES["Genus"]].index
    df = df[df['genus'].isin(valid_genera)].copy()
    return df


def create_cofgs_dataset(df_cofg):
    """Create dataset for H-COFGS and F-S (Species level)"""
    df = df_cofg.copy()
    df = df[df['species'].notna() & (df['species'] != '')].copy()
    df = remove_uncertain_species(df)
    species_counts = df['species'].value_counts()
    valid_species = species_counts[species_counts >= MIN_SAMPLES["Species"]].index
    df = df[df['species'].isin(valid_species)].copy()
    return df


def create_model_mapping():
    """Create model_data_mapping.json file"""
    mapping = {
        "F-C": "../cleaned/labels_clean.csv",  # Uses cleaned data without filtering
        "H-CO": "labels_clean_CO.csv",
        "H-COF": "labels_clean_COF.csv",
        "H-COFG": "labels_clean_COFG.csv",
        "F-G": "labels_clean_COFG.csv",
        "H-COFGS": "labels_clean_COFGS.csv",
        "F-S": "labels_clean_COFGS.csv"
    }
    
    mapping_file = DATA_ROOT / "preprocessed" / "model_data_mapping.json"
    with open(mapping_file, 'w', encoding='utf-8') as f:
        json.dump(mapping, f, indent=2, ensure_ascii=False)
    
    return mapping


def main():
    # Check if input file exists
    if not LABELS_CLEAN_CSV.exists():
        print(f"Error: Input file not found: {LABELS_CLEAN_CSV}")
        print("Please run python -m scripts.data.cleaning.clean_data first!")
        sys.exit(1)
    
    # Load base cleaned data
    df_base = pd.read_csv(LABELS_CLEAN_CSV)
    
    # Verify required columns exist
    required_cols = ['class', 'order', 'family', 'genus']
    missing_cols = [col for col in required_cols if col not in df_base.columns]
    if missing_cols:
        print(f"Error: Missing required columns: {missing_cols}")
        sys.exit(1)
    
    # Create progressive filtered datasets
    df_co = create_co_dataset(df_base)
    df_cof = create_cof_dataset(df_co)
    df_cofg = create_cofg_dataset(df_cof)
    df_cofgs = create_cofgs_dataset(df_cofg)
    
    # Check for potential issues
    if len(df_co) == 0:
        print("Warning: H-CO dataset is empty!")
    if len(df_cof) == 0:
        print("Warning: H-COF dataset is empty!")
    if len(df_cofg) == 0:
        print("Warning: H-COFG dataset is empty!")
    if len(df_cofgs) == 0:
        print("Warning: H-COFGS dataset is empty!")
    
    # Check if filtering removed too much data
    if len(df_co) < len(df_base) * 0.5:
        print(f"Warning: H-CO filtering removed >50% of data ({len(df_base)} -> {len(df_co)})")
    if len(df_cof) < len(df_co) * 0.5:
        print(f"Warning: H-COF filtering removed >50% of data ({len(df_co)} -> {len(df_cof)})")
    if len(df_cofg) < len(df_cof) * 0.5:
        print(f"Warning: H-COFG filtering removed >50% of data ({len(df_cof)} -> {len(df_cofg)})")
    if len(df_cofgs) < len(df_cofg) * 0.5:
        print(f"Warning: H-COFGS filtering removed >50% of data ({len(df_cofg)} -> {len(df_cofgs)})")
    
    # Save datasets
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_files = {
        "labels_clean_CO.csv": df_co,
        "labels_clean_COF.csv": df_cof,
        "labels_clean_COFG.csv": df_cofg,
        "labels_clean_COFGS.csv": df_cofgs
    }
    
    for filename, df in output_files.items():
        output_path = OUTPUT_DIR / filename
        df.to_csv(output_path, index=False)
    
    # Create model mapping
    create_model_mapping()
    
    # Summary
    print(f"Dataset creation complete: Base {len(df_base)}, H-CO {len(df_co)}, H-COF {len(df_cof)}, H-COFG {len(df_cofg)}, H-COFGS {len(df_cofgs)}")


if __name__ == "__main__":
    main()

