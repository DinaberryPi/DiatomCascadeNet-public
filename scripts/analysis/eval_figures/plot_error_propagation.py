#!/usr/bin/env python3
"""
Error Propagation Comparison: H-COFGS vs F-S
Plot error propagation analysis results
"""

import sys
from pathlib import Path

from scripts.analysis.eval_figures._common import *
import json
import numpy as np
import argparse


def plot_error_propagation_comparison():
    """Plot error propagation comparison from JSON results."""
    print("\n[INFO] Generating error propagation comparison...")
    
    # Load results
    results_file = REPORT_DIR / "error_propagation" / "error_propagation_H_COFGS_vs_F_S_results.json"
    
    if not results_file.exists():
        print(f"❌ Error propagation results file not found: {results_file}")
        print("   Please run: python experiment/error_propagation/error_propagation_H_COFGS_vs_F_S.py")
        return
    
    with open(results_file, 'r', encoding='utf-8') as f:
        results = json.load(f)
    
    h_cofgs_results = results.get('H-COFGS', {})
    f_s_results = results.get('F-S', {})
    
    if not h_cofgs_results or not f_s_results:
        print("❌ Missing results data")
        return
    
    # Create figure with better proportions
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 8))
    
    # Color scheme matching the visual style: light teal for H-COFGS, light salmon for F-S
    color_h_cofgs = '#2AB7CA'  # Light teal/aqua green for H-COFGS (Hierarchical)
    color_f_s = '#FFA07A'      # Light salmon/coral red for F-S (Lookup)
    
    # === Panel A: Upper-level correctness ===
    levels = ['Genus', 'Family', 'Order', 'Class']
    h_vals = [
        h_cofgs_results.get('genus_correct_given_species_error', 0),
        h_cofgs_results.get('family_correct_given_species_error', 0),
        h_cofgs_results.get('order_correct_given_species_error', 0),
        h_cofgs_results.get('class_correct_given_species_error', 0)
    ]
    f_vals = [
        f_s_results.get('genus_correct_given_species_error', 0),
        f_s_results.get('family_correct_given_species_error', 0),
        f_s_results.get('order_correct_given_species_error', 0),
        f_s_results.get('class_correct_given_species_error', 0)
    ]
    
    x = np.arange(len(levels))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, h_vals, width, 
                   label='H-COFGS (Hierarchical)', 
                   color=color_h_cofgs, alpha=0.85, 
                   edgecolor='black', linewidth=0.7)
    bars2 = ax1.bar(x + width/2, f_vals, width, 
                   label='F-S (Lookup)', 
                   color=color_f_s, alpha=0.85, 
                   edgecolor='black', linewidth=0.7)
    
    # Styling
    ax1.set_ylabel('Correctness Rate', fontsize=12, fontweight='bold')
    ax1.set_title('(A) Upper-Level Correctness Given Species Error', 
                 fontsize=13, fontweight='bold', loc='left', pad=15)
    ax1.set_xticks(x)
    ax1.set_xticklabels(levels, fontsize=11)
    ax1.set_ylim([0, 1.05])
    
    # Clean grid
    ax1.yaxis.grid(True, alpha=0.2, linestyle='-', linewidth=0.5, color='gray')
    ax1.set_axisbelow(True)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    # Value labels - cleaner style
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1%}',
                    ha='center', va='bottom', 
                    fontsize=9, color='black')
    
    # Legend inside plot
    ax1.legend(loc='lower right', fontsize=10, frameon=True, 
              edgecolor='black', framealpha=0.95)
    
    # === Panel B: Taxonomic distance distribution ===
    distances = [1, 2, 3, 4, 5]
    h_dist_counts = [h_cofgs_results.get('taxonomic_distance_distribution', {}).get(str(d), 0) 
                     for d in distances]
    f_dist_counts = [f_s_results.get('taxonomic_distance_distribution', {}).get(str(d), 0) 
                     for d in distances]
    
    h_species_errors = h_cofgs_results.get('species_errors', 1)
    f_species_errors = f_s_results.get('species_errors', 1)
    
    h_dist_pcts = [c / h_species_errors for c in h_dist_counts]
    f_dist_pcts = [c / f_species_errors for c in f_dist_counts]
    
    x = np.arange(len(distances))
    bars1 = ax2.bar(x - width/2, h_dist_pcts, width, 
                   label='H-COFGS (Hierarchical)', 
                   color=color_h_cofgs, alpha=0.85, 
                   edgecolor='black', linewidth=0.7)
    bars2 = ax2.bar(x + width/2, f_dist_pcts, width, 
                   label='F-S (Lookup)', 
                   color=color_f_s, alpha=0.85, 
                   edgecolor='black', linewidth=0.7)
    
    # Styling
    ax2.set_xlabel('Taxonomic Distance', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Proportion of Errors', fontsize=12, fontweight='bold')
    ax2.set_title('(B) Distribution of Error Distances', 
                 fontsize=13, fontweight='bold', loc='left', pad=15)
    ax2.set_xticks(x)
    ax2.set_xticklabels(['1\n(Genus)', '2\n(Family)', 
                        '3\n(Order)', '4\n(Class)', 
                        '5\n(Diff Class)'], fontsize=10)
    
    max_pct = max(max(h_dist_pcts) if h_dist_pcts else [0], 
                  max(f_dist_pcts) if f_dist_pcts else [0])
    ax2.set_ylim([0, max_pct * 1.15])
    
    # Clean grid
    ax2.yaxis.grid(True, alpha=0.2, linestyle='-', linewidth=0.5, color='gray')
    ax2.set_axisbelow(True)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    
    # Value labels - only show values > 1%
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if height > 0.01:
                ax2.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.1%}',
                        ha='center', va='bottom', 
                        fontsize=9, color='black')
    
    # Legend
    ax2.legend(loc='upper right', fontsize=10, frameon=True, 
              edgecolor='black', framealpha=0.95)
    
    # Layout
    plt.tight_layout(pad=1.5)
    
    # Save
    output_path = REPORT_DIR / "error_propagation" / "error_propagation_H_COFGS_vs_F_S.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"[INFO] Saved: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate error propagation comparison figure')
    args = parser.parse_args()
    
    try:
        plot_error_propagation_comparison()
    except Exception as e:
        print(f"[ERROR] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)