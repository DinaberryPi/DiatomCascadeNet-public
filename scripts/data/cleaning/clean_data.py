#!/usr/bin/env python3
"""
Data Cleaning Script for DiatomScanNet

This script:
1. Loads raw labels.csv
2. Merges Mediophyceae into Coscinodiscophyceae (insufficient samples)
3. Removes samples with missing labels at any taxonomic level
4. Removes uncertain/unknown species (sus, sp., aff., etc.) - not valid taxonomic names
5. Fixes taxonomic inconsistencies:
   - species->genus: FILTERS OUT ambiguous cases (same species epithet with multiple genera)
     Reason: In binomial nomenclature, the same epithet can appear in different genera
     (e.g., Navicula radiosa vs Pinnularia radiosa). However, if this occurs in our dataset,
     it could be annotation error. To be safe, we filter these out rather than fixing them.
   - genus->family, family->order, order->class: FIXES inconsistencies using majority vote
6. Saves cleaned data to labels_clean.csv

Usage:
    python -m scripts.data.cleaning.clean_data
"""

import hashlib
import re
import sys
from pathlib import Path

import pandas as pd

from diatom_cascade.config.data_config import EXCLUSION_MANIFEST_SHA256
from diatom_cascade.config.path_config import get_project_root

PROJECT_ROOT = get_project_root()

# Paths
RAW_LABELS = Path("dataset/raw/labels.csv")  # Generated from raw metadata
CLEAN_LABELS = Path("dataset/cleaned/labels_clean.csv")  # Cleaned for training
INVALID_IMAGES = Path("dataset/exclusions/invalid_images.csv")

def main():
    # Load raw data
    df = pd.read_csv(RAW_LABELS)

    if not INVALID_IMAGES.is_file():
        raise FileNotFoundError(f"Required exclusion manifest not found: {INVALID_IMAGES}")
    manifest_hash = hashlib.sha256(INVALID_IMAGES.read_bytes()).hexdigest()
    if manifest_hash != EXCLUSION_MANIFEST_SHA256:
        raise ValueError(
            f"Exclusion manifest hash mismatch: expected {EXCLUSION_MANIFEST_SHA256}, got {manifest_hash}"
        )
    exclusions = pd.read_csv(INVALID_IMAGES)
    if exclusions.empty or "filename" not in exclusions.columns or exclusions["filename"].duplicated().any():
        raise ValueError(f"Invalid exclusion manifest: {INVALID_IMAGES}")
    excluded_names = set(exclusions["filename"].astype(str))
    unknown_names = excluded_names - set(df["filename"].astype(str))
    if unknown_names:
        raise ValueError(f"Exclusion manifest contains unknown filenames: {sorted(unknown_names)}")
    df = df[~df["filename"].astype(str).isin(excluded_names)].copy()
    print(f"Excluded {len(excluded_names)} invalid images from {INVALID_IMAGES}")
    
    # Step 1: Merge Mediophyceae into Coscinodiscophyceae
    merged_count = (df['class'] == 'Mediophyceae').sum()
    if merged_count > 0:
        df.loc[df['class'] == 'Mediophyceae', 'class'] = 'Coscinodiscophyceae'
    
    # Step 2: Remove samples with missing labels
    before_filter = len(df)
    df = df[
        (df['class'].notna() & (df['class'] != '')) &
        (df['order'].notna() & (df['order'] != '')) &
        (df['family'].notna() & (df['family'] != ''))
    ].copy()
    
    # Step 3: Remove uncertain species (sus, sp., aff., etc.)
    
    # Define uncertain species markers
    # These markers indicate incomplete or uncertain identifications and are NOT valid taxonomic names
    # according to the International Code of Nomenclature (ICN).
    #
    # Scientific basis for removal:
    # - "sus" = suspected/unknown species (not a valid taxonomic name)
    # - "sp." = species (uncertain species, often used when exact species is unknown)
    #   Example: "Navicula sp." = "some Navicula species, but which one is unknown"
    # - "aff." = affinis (similar to but not confirmed as a species)
    #   Example: "Navicula aff. radiosa" = "similar to N. radiosa but not confirmed"
    # - "cf." = confer (compare, used for uncertain identifications)
    #   Example: "Cyclotella cf. meneghiniana" = "compare with C. meneghiniana"
    # - "species" = generic "species" word (uncertain, like "Cyclotella species")
    #   Example: "Cyclotella species" = "some Cyclotella species, but which one is unknown"
    #   This is NOT a valid taxonomic name. Valid names follow binomial nomenclature:
    #   "Cyclotella meneghiniana" (valid) vs "Cyclotella species" (invalid)
    #   NOTE: "species" is handled separately below because it must match only at the END of the string
    #   to avoid false positives (e.g., "species_name" should NOT be filtered)
    #
    # Reference: International Code of Nomenclature (ICN) Article 23.1-23.2
    # See docs/data_cleaning/UNCERTAIN_SPECIES_CRITERIA.md for detailed explanation
    uncertain_markers = ['sus', 'sp.', 'aff.', 'cf.', '?', 'unknown', 'undetermined', 'indet']
    # Note: 'species' is NOT in this list because it requires special end-of-string matching
    # (see separate handling below at line ~159)
    
    # Check for uncertain markers in species and genus columns
    uncertain_mask = pd.Series([False] * len(df), index=df.index)
    
    for col in ['species', 'genus']:
        for marker in uncertain_markers:
            # Use regex with word boundaries to avoid false matches (e.g., "affine" matching "aff.")
            if marker in ['sp.', 'aff.', 'cf.']:
                # Match as word boundary pattern, but allow end of string after period
                # Use negative lookahead to ensure not followed by word character
                # Note: re.escape already escapes the period, so we don't need to replace it
                pattern = r'\b' + re.escape(marker) + r'(?!\w)'
                mask = df[col].astype(str).str.contains(pattern, case=False, na=False, regex=True)
            else:
                # For other markers, use word boundaries
                pattern = r'\b' + re.escape(marker) + r'\b'
                mask = df[col].astype(str).str.contains(pattern, case=False, na=False, regex=True)
            uncertain_mask |= mask
        
        # Also check for "species" as a word at the END of the string (e.g., "Cyclotella species")
        # "species" is NOT a valid taxonomic name - it's just a generic term meaning "some species"
        # Valid names must follow binomial nomenclature: "Genus specific_epithet"
        # Example: "Cyclotella species" (invalid) vs "Cyclotella meneghiniana" (valid)
        # 
        # Why separate handling? We only match "species" at the END to avoid false positives:
        #
        # Excel 示例对比：
        # ┌─────────────────────┬──────────────────┬──────────┬─────────────────────┐
        # │ species 列的值      │ 如果匹配任何位置  │ 是否正确 │ 如果只匹配末尾      │
        # ├─────────────────────┼──────────────────┼──────────┼─────────────────────┤
        # │ "species"           │ ✅ 匹配           │ ✅ 正确  │ ✅ 匹配（末尾）     │
        # │ "species_name"      │ ✅ 匹配 ❌ 错误！ │ ❌ 误过滤│ ❌ 不匹配 ✅ 正确   │
        # │ "some_species_var"  │ ✅ 匹配 ❌ 错误！ │ ❌ 误过滤│ ❌ 不匹配 ✅ 正确   │
        # │ "meneghiniana"      │ ❌ 不匹配         │ ✅ 正确  │ ❌ 不匹配 ✅ 正确   │
        # └─────────────────────┴──────────────────┴──────────┴─────────────────────┘
        #
        # 结论：'species' 必须在末尾匹配，否则会误过滤有效名称！
        species_pattern = r'\bspecies\b$'
        mask = df[col].astype(str).str.contains(species_pattern, case=False, na=False, regex=True)
        uncertain_mask |= mask
    
    uncertain_count = uncertain_mask.sum()
    
    if uncertain_count > 0:
        uncertain_samples = df[uncertain_mask].copy()
        
        # Save uncertain samples to Excel for review
        uncertain_excel_path = Path("dataset/cleaned/uncertain_species.xlsx")
        export_cols = ['filename', 'class', 'order', 'family', 'genus', 'species']
        export_cols = [col for col in export_cols if col in uncertain_samples.columns]
        uncertain_samples[export_cols].to_excel(uncertain_excel_path, index=False, engine='openpyxl')
        
        df = df[~uncertain_mask].copy()
    
    # Step 4: Fix taxonomic inconsistencies (all levels)
    from collections import defaultdict
    
    total_fixed = 0
    total_filtered = 0
    inconsistency_records = []
    
    # 4.1: Check species -> genus consistency
    species_to_genera = defaultdict(set)
    for _, row in df.iterrows():
        species = str(row.get('species', '')).strip()
        genus = str(row.get('genus', '')).strip()
        if species and genus and species != 'nan' and genus != 'nan':
            species_to_genera[species].add((genus, row.get('family', ''), row.get('order', ''), row.get('class', '')))
    
    inconsistent_species = {s: genera for s, genera in species_to_genera.items() if len(genera) > 1}
    
    if inconsistent_species:
        samples_to_filter = []
        for species, genera in inconsistent_species.items():
            species_mask = df['species'] == species
            samples_to_filter.extend(df[species_mask].index.tolist())
        
        filtered_species_count = len(samples_to_filter)
        df = df.drop(index=samples_to_filter).reset_index(drop=True)
        total_filtered += filtered_species_count
        if filtered_species_count > 0:
            print(f"Filtered {filtered_species_count} samples with ambiguous species->genus relationships")
    
    # 4.2: Check genus -> family consistency
    genus_to_families = defaultdict(set)
    for _, row in df.iterrows():
        genus_to_families[row['genus']].add((row['family'], row['order'], row['class']))
    
    inconsistent_genera = {g: families for g, families in genus_to_families.items() if len(families) > 1}
    
    if inconsistent_genera:
        fixed_genus_count = 0
        for genus, families in inconsistent_genera.items():
            genus_df = df[df['genus'] == genus]
            correct_family = genus_df['family'].value_counts().index[0]
            correct_order = genus_df[genus_df['family'] == correct_family]['order'].iloc[0]
            correct_class = genus_df[genus_df['family'] == correct_family]['class'].iloc[0]
            
            wrong_family_mask = (df['genus'] == genus) & (df['family'] != correct_family)
            num_to_fix = wrong_family_mask.sum()
            if num_to_fix > 0:
                df.loc[wrong_family_mask, 'family'] = correct_family
                df.loc[wrong_family_mask, 'order'] = correct_order
                df.loc[wrong_family_mask, 'class'] = correct_class
                fixed_genus_count += num_to_fix
        
        total_fixed += fixed_genus_count
        if fixed_genus_count > 0:
            print(f"Fixed {fixed_genus_count} samples with inconsistent genus->family relationships")
    
    # 4.3: Check family -> order consistency
    family_to_orders = defaultdict(set)
    for _, row in df.iterrows():
        family_to_orders[row['family']].add((row['order'], row['class']))
    
    inconsistent_families = {f: orders for f, orders in family_to_orders.items() if len(orders) > 1}
    
    if inconsistent_families:
        fixed_family_count = 0
        for family, orders in inconsistent_families.items():
            family_df = df[df['family'] == family]
            correct_order = family_df['order'].value_counts().index[0]
            correct_class = family_df[family_df['order'] == correct_order]['class'].iloc[0]
            
            wrong_order_mask = (df['family'] == family) & (df['order'] != correct_order)
            num_to_fix = wrong_order_mask.sum()
            if num_to_fix > 0:
                df.loc[wrong_order_mask, 'order'] = correct_order
                df.loc[wrong_order_mask, 'class'] = correct_class
                fixed_family_count += num_to_fix
        
        total_fixed += fixed_family_count
        if fixed_family_count > 0:
            print(f"Fixed {fixed_family_count} samples with inconsistent family->order relationships")
    
    # 4.4: Check order -> class consistency
    order_to_classes = defaultdict(set)
    for _, row in df.iterrows():
        order_to_classes[row['order']].add(row['class'])
    
    inconsistent_orders = {o: classes for o, classes in order_to_classes.items() if len(classes) > 1}
    
    if inconsistent_orders:
        fixed_order_count = 0
        for order, classes in inconsistent_orders.items():
            order_df = df[df['order'] == order]
            correct_class = order_df['class'].value_counts().index[0]
            
            wrong_class_mask = (df['order'] == order) & (df['class'] != correct_class)
            num_to_fix = wrong_class_mask.sum()
            if num_to_fix > 0:
                df.loc[wrong_class_mask, 'class'] = correct_class
                fixed_order_count += num_to_fix
        
        total_fixed += fixed_order_count
        if fixed_order_count > 0:
            print(f"Fixed {fixed_order_count} samples with inconsistent order->class relationships")
    
    # Save cleaned data
    CLEAN_LABELS.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(CLEAN_LABELS, index=False)
    
    # Check for potential issues
    if len(df) == 0:
        print("Error: No samples remaining after cleaning!")
        sys.exit(1)
    
    if len(df) < before_filter * 0.3:
        print(f"Warning: Cleaning removed >70% of data ({before_filter} -> {len(df)})")
    
    print(f"Data cleaning complete: {len(df)} samples | Classes: {df['class'].nunique()} | Orders: {df['order'].nunique()} | "
          f"Families: {df['family'].nunique()} | Genera: {df['genus'].nunique()} | Species: {df['species'].nunique()}")
    
    if total_filtered > 0 or total_fixed > 0:
        print(f"Filtered: {total_filtered} samples | Fixed: {total_fixed} samples")

if __name__ == "__main__":
    main()

