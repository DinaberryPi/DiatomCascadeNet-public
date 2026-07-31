#!/usr/bin/env python3
"""
Pipeline Stage Pyramids Visualization
Display taxonomy pyramids for different stages of the data pipeline:
- Raw data
- Cleaned data
- Filtered datasets (CO, COF, COFG, COFGS)
"""

import sys
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
import argparse

from diatom_cascade.config.data_config import MIN_SAMPLES

# Set style
plt.rcParams['font.size'] = 10
plt.rcParams['font.family'] = 'sans-serif'

# Paths
RAW_LABELS = Path("dataset/raw/labels.csv")
CLEAN_LABELS = Path("dataset/cleaned/labels_clean.csv")
PREPROCESSED_DIR = Path("dataset/preprocessed")
REPORT_DIR = Path("report")
REPORT_DIR.mkdir(parents=True, exist_ok=True)

# Pipeline stages - matching original format exactly
PIPELINE_STAGES = {
    'Raw': {
        'path': RAW_LABELS,
        'filter': None,
        'model': None
    },
    'Cleaned': {
        'path': CLEAN_LABELS,
        'filter': 'Remove missing/ambiguous',
        'model': 'F-C'
    },
    'Cleaned & Filtered, H-CO': {
        'path': PREPROCESSED_DIR / "labels_clean_CO.csv",
        'filter': f'O\u2265{MIN_SAMPLES["Order"]}',
        'model': 'H-CO'
    },
    'Cleaned & Filtered, H-COF': {
        'path': PREPROCESSED_DIR / "labels_clean_COF.csv",
        'filter': f'O\u2265{MIN_SAMPLES["Order"]}, F\u2265{MIN_SAMPLES["Family"]}',
        'model': 'H-COF'
    },
    'Cleaned & Filtered, H-COFG': {
        'path': PREPROCESSED_DIR / "labels_clean_COFG.csv",
        'filter': f'O\u2265{MIN_SAMPLES["Order"]}, F\u2265{MIN_SAMPLES["Family"]}, G\u2265{MIN_SAMPLES["Genus"]}',
        'model': 'H-COFG'
    },
    'Cleaned & Filtered, H-COFGS': {
        'path': PREPROCESSED_DIR / "labels_clean_COFGS.csv",
        'filter': f'O\u2265{MIN_SAMPLES["Order"]}, F\u2265{MIN_SAMPLES["Family"]}, G\u2265{MIN_SAMPLES["Genus"]}, S\u2265{MIN_SAMPLES["Species"]}',
        'model': 'H-COFGS, F-S'
    }
}

# Dark teal color used throughout
DARK_TEAL = '#1A5276'


def load_taxonomy_counts(file_path):
    """Load taxonomy data and count unique taxa at each level"""
    if not file_path.exists():
        return None
    
    try:
        df = pd.read_csv(file_path)
        
        counts = {}
        if 'class' in df.columns:
            counts['Class'] = df['class'].nunique()
        if 'order' in df.columns:
            counts['Order'] = df['order'].nunique()
        if 'family' in df.columns:
            counts['Family'] = df['family'].nunique()
        if 'genus' in df.columns:
            counts['Genus'] = df['genus'].nunique()
        if 'species' in df.columns:
            counts['Species'] = df['species'].nunique()
        
        return counts, len(df)
    except Exception as e:
        print(f"Warning: Could not load {file_path}: {e}")
        return None


def plot_single_pyramid(ax, counts, stage_name, total_samples, filter_criteria=None, max_count_all=None, model_name=None):
    """Plot a single pyramid on given axes"""
    levels = ['Class', 'Order', 'Family', 'Genus', 'Species']
    level_abbrev = {'Class': 'C', 'Order': 'O', 'Family': 'F', 'Genus': 'G', 'Species': 'S'}
    level_counts = [counts.get(level, 0) for level in levels]
    
    # Color scheme
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6A994E']
    
    # Use provided max_count or calculate from current data
    if max_count_all is None:
        max_count = max(level_counts) if level_counts else 1
    else:
        max_count = max_count_all
    
    if max_count == 0:
        max_count = 1
    
    num_levels = len(levels)
    bar_heights = 0.085
    spacing = 0.085
    
    for i, (level, count) in enumerate(zip(levels, level_counts)):
        if count == 0:
            continue
        
        width = np.sqrt(count / max_count) * 0.62 if max_count > 0 else 0
        y_pos = 0.68 - (i * spacing)
        x_center = 0.5
        x_left = x_center - width / 2
        
        rect = Rectangle((x_left, y_pos - bar_heights/2), width, bar_heights,
                        facecolor=colors[i], edgecolor='white', linewidth=1.5,
                        alpha=0.95, zorder=num_levels - i)
        ax.add_patch(rect)
        
        # Abbreviated label (C:, O:, F:, G:, S:)
        ax.text(0.02, y_pos, f'{level_abbrev[level]}:', ha='left', va='center',
               fontsize=9, fontweight='bold', color=DARK_TEAL)
        
        # Count label
        if level == 'Species':
            ax.text(x_center + width/2 + 0.02, y_pos, f'{count}', 
                    ha='left', va='center', fontsize=7.8, fontweight='bold',
                    color=colors[i])
        elif width > 0.18:
            ax.text(x_center, y_pos, f'{count}', ha='center', va='center',
                    fontsize=7.8, fontweight='bold', color='white')
        else:
            ax.text(x_center + width/2 + 0.02, y_pos, f'{count}', 
                    ha='left', va='center', fontsize=7.8, fontweight='bold',
                    color=colors[i])
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    # Title
    ax.text(0.5, 0.98, stage_name, 
           ha='center', va='top', fontsize=9, fontweight='bold',
           color=DARK_TEAL)
    
    # Sample count: n = X,XXX
    ax.text(0.5, 0.91, f'n = {total_samples:,}',
           ha='center', va='top', fontsize=8,
           color=DARK_TEAL)
    
    # Filter criteria
    if filter_criteria:
        ax.text(0.5, 0.84, filter_criteria,
               ha='center', va='top', fontsize=6.5, style='italic',
               color='#666666')
    
    # "Used by model" box
    if model_name:
        last_bar_y = 0.68 - (4 * spacing) - bar_heights/2
        box_y = last_bar_y - 0.08
        
        txt = ax.text(0.5, box_y, f'Used by model: {model_name}',
               ha='center', va='center', fontsize=8, fontweight='bold',
               color=DARK_TEAL)
        
        bbox_props = dict(boxstyle="round,pad=0.15,rounding_size=0.2", 
                         facecolor='#E8F2F7', edgecolor=DARK_TEAL, linewidth=1.2)
        txt.set_bbox(bbox_props)


def plot_pipeline_pyramids():
    """Create pyramid visualizations for all pipeline stages"""
    print("[INFO] Generating Pipeline Stage Pyramids...")
    
    stage_data = {}
    all_counts = []
    
    for stage_name, stage_info in PIPELINE_STAGES.items():
        file_path = stage_info['path']
        filter_criteria = stage_info.get('filter')
        model_name = stage_info.get('model')
        result = load_taxonomy_counts(file_path)
        if result is not None:
            counts, total_samples = result
            stage_data[stage_name] = {
                'counts': counts,
                'total_samples': total_samples,
                'filter': filter_criteria,
                'model': model_name
            }
            all_counts.extend(counts.values())
    
    if not stage_data:
        print("[ERROR] No data files found!")
        return
    
    max_count_all = max(all_counts) if all_counts else 1
    
    cols = 3
    rows = 2
    fig = plt.figure(figsize=(7.16, 4.5))  # IEEE double-column width, compact
    
    for idx, (stage_name, data) in enumerate(stage_data.items(), 1):
        ax = fig.add_subplot(rows, cols, idx)
        plot_single_pyramid(ax, data['counts'], stage_name, 
                           data['total_samples'], data['filter'], max_count_all, data.get('model'))
    
    # Bottom legend - two lines, italic
    legend_line1 = 'C=Class  O=Order  F=Family  G=Genus  S=Species'
    legend_line2 = 'Bar width \u221d \u221a(taxa count)  |  Bottom box = model trained on this dataset'
    
    fig.text(0.5, 0.01, legend_line1 + '\n' + legend_line2,
            ha='center', va='bottom', fontsize=7, style='italic',
            color=DARK_TEAL, linespacing=1.2)
    
    plt.tight_layout(rect=[0, 0.055, 1, 0.99])
    plt.subplots_adjust(hspace=0.15, wspace=0.12)
    
    output_path = REPORT_DIR / 'taxonomy_pyramids.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"[INFO] Saved: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate pyramid visualizations')
    args = parser.parse_args()
    
    try:
        plot_pipeline_pyramids()
    except Exception as e:
        print(f"[ERROR] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
