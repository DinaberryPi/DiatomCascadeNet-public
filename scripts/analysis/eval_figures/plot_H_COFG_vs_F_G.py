#!/usr/bin/env python3
"""
H-COFG vs F-G Comparison
Compare 4 prediction methods: level-wise argmax (no constraints), greedy hierarchical (mask constraints), beam search, flat bottom-up lookup
For F-G: argmax does not look up upper levels; other methods use lookup
"""

import sys
from pathlib import Path

from scripts.analysis.eval_figures._common import *
import argparse


def plot_h_cofg_vs_f_g_comparison():
    """
    对比 H-COFG 与 F-G 的 4 种预测方式（统一画在一张图上）
    """
    print("\n📊 Generating H-COFG vs F-G comparison (4 methods)...")
    
    h_cofg_file = EVAL_DIR / 'H_COFG_evaluation_report.json'
    f_g_file = EVAL_DIR / 'F_G_evaluation_report.json'
    
    if not h_cofg_file.exists():
        print(f"❌ H-COFG evaluation file not found: {h_cofg_file}")
        return
    
    if not f_g_file.exists():
        print(f"❌ F-G evaluation file not found: {f_g_file}")
        return
    
    # Load results
    with open(h_cofg_file, 'r') as f:
        h_cofg_data = json.load(f)
    
    with open(f_g_file, 'r') as f:
        f_g_data = json.load(f)
    
    levels = ['Class', 'Order', 'Family', 'Genus']
    methods = ['argmax', 'greedy', 'beam_search', 'flat_lookup']
    method_labels = {
        'argmax': 'Level-wise Argmax (No Constraints)',
        'greedy': 'Greedy Hierarchical (Mask Constraints)',
        'beam_search': 'Beam Search',
        'flat_lookup': 'Flat Bottom-up Lookup'
    }
    
    # Extract H-COFG metrics for 3 methods
    h_cofg_metrics = {method: {} for method in ['argmax', 'greedy', 'beam_search']}
    
    # Argmax
    if 'argmax_prediction' in h_cofg_data:
        argmax_data = h_cofg_data['argmax_prediction']
    else:
        argmax_data = h_cofg_data
    
    h_cofg_metrics['argmax']['Class'] = (argmax_data.get('class_level', {}).get('accuracy', 0),
                                         argmax_data.get('class_level', {}).get('f1_weighted', 0))
    h_cofg_metrics['argmax']['Order'] = (argmax_data.get('order_level', {}).get('accuracy', 0),
                                         argmax_data.get('order_level', {}).get('f1_weighted', 0))
    h_cofg_metrics['argmax']['Family'] = (argmax_data.get('family_level', {}).get('accuracy', 0),
                                           argmax_data.get('family_level', {}).get('f1_weighted', 0))
    h_cofg_metrics['argmax']['Genus'] = (argmax_data.get('genus_level', {}).get('accuracy', 0),
                                          argmax_data.get('genus_level', {}).get('f1_weighted', 0))
    
    # Greedy
    if 'hierarchical_prediction' in h_cofg_data:
        hier_data = h_cofg_data['hierarchical_prediction']
    else:
        hier_data = h_cofg_data
    
    h_cofg_metrics['greedy']['Class'] = (hier_data.get('class_level', {}).get('accuracy', 0),
                                         hier_data.get('class_level', {}).get('f1_weighted', 0))
    h_cofg_metrics['greedy']['Order'] = (hier_data.get('order_level', {}).get('accuracy', 0),
                                         hier_data.get('order_level', {}).get('f1_weighted', 0))
    h_cofg_metrics['greedy']['Family'] = (hier_data.get('family_level', {}).get('accuracy', 0),
                                          hier_data.get('family_level', {}).get('f1_weighted', 0))
    h_cofg_metrics['greedy']['Genus'] = (hier_data.get('genus_level', {}).get('accuracy', 0),
                                         hier_data.get('genus_level', {}).get('f1_weighted', 0))
    
    # Beam Search
    if 'beam_search_prediction' in h_cofg_data:
        beam_data = h_cofg_data['beam_search_prediction']
    else:
        beam_data = hier_data
    
    h_cofg_metrics['beam_search']['Class'] = (beam_data.get('class_level', {}).get('accuracy', 0),
                                               beam_data.get('class_level', {}).get('f1_weighted', 0))
    h_cofg_metrics['beam_search']['Order'] = (beam_data.get('order_level', {}).get('accuracy', 0),
                                               beam_data.get('order_level', {}).get('f1_weighted', 0))
    h_cofg_metrics['beam_search']['Family'] = (beam_data.get('family_level', {}).get('accuracy', 0),
                                                beam_data.get('family_level', {}).get('f1_weighted', 0))
    h_cofg_metrics['beam_search']['Genus'] = (beam_data.get('genus_level', {}).get('accuracy', 0),
                                               beam_data.get('genus_level', {}).get('f1_weighted', 0))
    
    # Extract F-G metrics
    f_g_metrics = {method: {} for method in methods}
    
    # F-G: Argmax (only genus, no lookup)
    genus_metrics = f_g_data.get('genus_metrics', {})
    f_g_metrics['argmax']['Genus'] = (genus_metrics.get('accuracy', 0),
                                       genus_metrics.get('f1_weighted', 0))
    f_g_metrics['argmax']['Class'] = (None, None)  # No lookup for argmax
    f_g_metrics['argmax']['Order'] = (None, None)
    f_g_metrics['argmax']['Family'] = (None, None)
    
    # F-G: Other methods (genus + lookup upper levels)
    upper_level_metrics = f_g_data.get('upper_level_metrics', {})
    
    if upper_level_metrics:
        for method in ['greedy', 'beam_search', 'flat_lookup']:
            f_g_metrics[method]['Genus'] = (genus_metrics.get('accuracy', 0),
                                             genus_metrics.get('f1_weighted', 0))
            f_g_metrics[method]['Class'] = (upper_level_metrics.get('class', {}).get('accuracy', 0),
                                              upper_level_metrics.get('class', {}).get('f1_weighted', 0))
            f_g_metrics[method]['Order'] = (upper_level_metrics.get('order', {}).get('accuracy', 0),
                                             upper_level_metrics.get('order', {}).get('f1_weighted', 0))
            f_g_metrics[method]['Family'] = (upper_level_metrics.get('family', {}).get('accuracy', 0),
                                               upper_level_metrics.get('family', {}).get('f1_weighted', 0))
    else:
        # Fallback
        for method in ['greedy', 'beam_search', 'flat_lookup']:
            f_g_metrics[method]['Genus'] = (genus_metrics.get('accuracy', 0),
                                             genus_metrics.get('f1_weighted', 0))
            for level in ['Class', 'Order', 'Family']:
                f_g_metrics[method][level] = (None, None)
    
    # Combine into single figure (two subplots)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('H-COFG vs F-G: Prediction Methods Comparison',
                 fontsize=16, fontweight='bold', y=0.995)

    x_positions = np.arange(len(levels))
    width = 0.18

    method_colors = {
        'argmax': '#FFD166',       # Level-wise argmax -> yellow
        'greedy': '#FF6FB5',       # Greedy hierarchical -> pink
        'beam_search': '#2AB7CA',
        'flat_lookup': '#9370DB'
    }

    for idx, method in enumerate(methods):
        offset = (idx - 1.5) * width
        label = (f'H-COFG {method_labels[method]}'
                 if method != 'flat_lookup'
                 else 'F-G Flat Lookup')

        if method == 'flat_lookup':
            acc_values = [f_g_metrics[method][level][0] or 0 for level in levels]
            f1_values = [f_g_metrics[method][level][1] or 0 for level in levels]
        else:
            acc_values = [h_cofg_metrics[method][level][0] or 0 for level in levels]
            f1_values = [h_cofg_metrics[method][level][1] or 0 for level in levels]

        bars_acc = ax1.bar(x_positions + offset, acc_values, width,
                           label=label, color=method_colors[method], alpha=0.85)
        bars_f1 = ax2.bar(x_positions + offset, f1_values, width,
                          label=label, color=method_colors[method], alpha=0.85)

        for bar, value in zip(bars_acc, acc_values):
            if value > 0:
                ax1.text(bar.get_x() + bar.get_width() / 2., value + 0.01,
                         f'{value:.3f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
        for bar, value in zip(bars_f1, f1_values):
            if value > 0:
                ax2.text(bar.get_x() + bar.get_width() / 2., value + 0.01,
                         f'{value:.3f}', ha='center', va='bottom', fontsize=8, fontweight='bold')

    ax1.set_title('Accuracy by Level', fontsize=14, fontweight='semibold')
    ax1.set_xlabel('Taxonomic Level', fontsize=12, fontweight='semibold')
    ax1.set_ylabel('Accuracy', fontsize=12, fontweight='semibold')
    ax1.set_xticks(x_positions)
    ax1.set_xticklabels(levels, fontsize=11)
    ax1.set_ylim(0, 1.05)
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.legend(fontsize=9, loc='lower left')

    ax2.set_title('Weighted F1 by Level', fontsize=14, fontweight='semibold')
    ax2.set_xlabel('Taxonomic Level', fontsize=12, fontweight='semibold')
    ax2.set_ylabel('Weighted F1', fontsize=12, fontweight='semibold')
    ax2.set_xticks(x_positions)
    ax2.set_xticklabels(levels, fontsize=11)
    ax2.set_ylim(0, 1.05)
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.legend(fontsize=9, loc='lower left')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    output_path = REPORT_DIR / 'H_COFG_vs_F_G_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate H-COFG vs F-G comparison with 4 methods')
    args = parser.parse_args()
    
    try:
        plot_h_cofg_vs_f_g_comparison()
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

