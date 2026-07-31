#!/usr/bin/env python3
"""
H-COFGS vs F-S Comparison
Compare different prediction methods between hierarchical and flat models
"""

import sys
from pathlib import Path

from scripts.analysis.eval_figures._common import *
import json
import numpy as np
import argparse


def plot_h_cofgs_vs_f_s_comparison(mode='full'):
    """
    对比 H-COFGS 与 F-S 的四种预测方式：
    3 个层级模型方法 + 1 个 flat bottom-up 方法
    """
    print(f"\n[INFO] Generating H-COFGS vs F-S comparison (mode={mode})...")

    h_cofgs_file = EVAL_DIR / 'H_COFGS_evaluation_report.json'
    f_s_file = EVAL_DIR / 'F_S_evaluation_report.json'

    if not h_cofgs_file.exists():
        print(f"❌ H-COFGS evaluation file not found: {h_cofgs_file}")
        return

    if not f_s_file.exists():
        print(f"❌ F-S evaluation file not found: {f_s_file}")
        return

    with open(h_cofgs_file, 'r') as f:
        h_cofgs_data = json.load(f)

    with open(f_s_file, 'r') as f:
        f_s_data = json.load(f)

    levels = ['Class', 'Order', 'Family', 'Genus', 'Species']
    if mode == 'minimal':
        methods = ['greedy', 'flat_lookup']
    else:
        methods = ['argmax', 'greedy', 'beam_search', 'flat_lookup']
    method_labels = {
        'argmax': 'Level-wise Argmax',  # 缩短标签
        'greedy': 'Greedy Hierarchical',
        'beam_search': 'Beam Search',
        'flat_lookup': 'Flat Bottom-up'
    }

    # Extract H-COFGS metrics
    h_metrics = {method: {} for method in ['argmax', 'greedy', 'beam_search']}
    argmax_data = h_cofgs_data.get('argmax_prediction', h_cofgs_data)
    hier_data = h_cofgs_data.get('hierarchical_prediction', h_cofgs_data)
    beam_data = h_cofgs_data.get('beam_search_prediction', hier_data)

    for level in levels:
        key = f"{level.lower()}_level"
        h_metrics['argmax'][level] = (
            argmax_data.get(key, {}).get('accuracy', 0),
            argmax_data.get(key, {}).get('f1_weighted', 0)
        )
        h_metrics['greedy'][level] = (
            hier_data.get(key, {}).get('accuracy', 0),
            hier_data.get(key, {}).get('f1_weighted', 0)
        )
        h_metrics['beam_search'][level] = (
            beam_data.get(key, {}).get('accuracy', 0),
            beam_data.get(key, {}).get('f1_weighted', 0)
        )

    # Extract F-S metrics (Flat)
    f_metrics = {}
    species_metrics = f_s_data.get('species_metrics', {})
    upper_level_metrics = f_s_data.get('upper_level_metrics', {})

    f_metrics['Species'] = (
        upper_level_metrics.get('species', species_metrics).get('accuracy', 0),
        upper_level_metrics.get('species', species_metrics).get('f1_weighted', 0)
    )

    for level in ['Class', 'Order', 'Family', 'Genus']:
        level_key = level.lower()
        if upper_level_metrics:
            f_metrics[level] = (
                upper_level_metrics.get(level_key, {}).get('accuracy', 0),
                upper_level_metrics.get(level_key, {}).get('f1_weighted', 0)
            )
        else:
            f_metrics[level] = (None, None)

    # 优化的图表设置 - 适合IEEE双栏格式
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(3.5, 5), sharex=True)
    
    # Remove overall suptitle - each subplot has its own title matching the image

    x_positions = np.arange(len(levels))
    width = 0.6 / len(methods)
    center_offset = (len(methods) - 1) / 2

    # Color scheme based on mode
    if mode == 'minimal':
        # Minimal mode: 2 methods - orange/peach for greedy, light blue for flat
        colors = {
            'greedy': '#FF8C42',      # Orange/peach for H-COFGS Greedy Hierarchical
            'flat_lookup': '#ADD8E6'   # Light blue/pastel blue for Flat & Upper Lookup
        }
    else:
        # Full mode: 4 methods
        colors = {
            'argmax': '#FF7F50',      # Coral/red-orange for Level-wise Argmax
            'greedy': '#FFD700',      # Gold for Greedy Hierarchical
            'beam_search': '#2AB7CA', # Light teal/cyan for Beam Search
            'flat_lookup': '#008B8B'   # Dark teal/blue-green for Flat & Upper Lookup
        }

    for idx, method in enumerate(methods):
        offset = (idx - center_offset) * width
        if method == 'flat_lookup':
            acc_values = [f_metrics[level][0] or 0 for level in levels]
            f1_values = [f_metrics[level][1] or 0 for level in levels]
            label = 'Flat & Upper Lookup'  # Match image label
        else:
            acc_values = [h_metrics[method].get(level, (None, None))[0] or 0 for level in levels]
            f1_values = [h_metrics[method].get(level, (None, None))[1] or 0 for level in levels]
            if mode == 'minimal':
                label = 'H-COFGS Greedy Hierarchical'  # Match image label
            else:
                label = f'H-COFGS {method_labels[method]}'

        bars_acc = ax1.bar(x_positions + offset, acc_values, width,
                           label=label, color=colors[method], alpha=0.85)
        bars_f1 = ax2.bar(x_positions + offset, f1_values, width,
                          label=label, color=colors[method], alpha=0.85)

        # 优化数值标注 - 匹配图片格式（3位小数）
        for bar, value in zip(bars_acc, acc_values):
            if value > 0:
                text_y = value + 0.02  # 稍微增加偏移
                ax1.text(bar.get_x() + bar.get_width() / 2., text_y,
                         f'{value:.3f}', ha='center', va='bottom',
                         fontsize=7, fontweight='normal', color='black')

        for bar, value in zip(bars_f1, f1_values):
            if value > 0:
                text_y = value + 0.02
                ax2.text(bar.get_x() + bar.get_width() / 2., text_y,
                         f'{value:.3f}', ha='center', va='bottom',
                         fontsize=7, fontweight='normal', color='black')

    # 调整所有文字大小以适配IEEE格式 - 匹配图片标题格式
    ax1.set_title('H-COFGS vs F-S: Accuracy by Level', fontsize=10, fontweight='semibold')
    ax1.set_ylabel('Accuracy', fontsize=9, fontweight='semibold')
    ax1.set_xticks(x_positions)
    ax1.set_xticklabels([])
    ax1.set_ylim(0, 1.10)  # 稍微降低上限
    ax1.tick_params(axis='y', labelsize=8)
    ax1.grid(True, alpha=0.3, axis='y', linewidth=0.5)
    ax1.legend(fontsize=7, loc='lower left', framealpha=0.9)

    ax2.set_title('H-COFGS vs F-S: Weighted F1 by Level', fontsize=10, fontweight='semibold')
    ax2.set_ylabel('Weighted F1', fontsize=9, fontweight='semibold')
    ax2.set_xticks(x_positions)
    ax2.set_xticklabels(levels, fontsize=8)
    ax2.set_ylim(0, 1.10)
    ax2.tick_params(axis='both', labelsize=8)
    ax2.grid(True, alpha=0.3, axis='y', linewidth=0.5)
    ax2.legend(fontsize=7, loc='lower left', framealpha=0.9)

    # 紧凑布局
    plt.tight_layout(rect=[0, 0, 1, 0.99], h_pad=1.5)
    
    output_name = 'H_COFGS_vs_F_S_comparison_minimal.png' if mode == 'minimal' else 'H_COFGS_vs_F_S_comparison.png'
    output_path = REPORT_DIR / output_name
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[INFO] Saved: {output_path}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate H-COFGS vs F-S comparison')
    parser.add_argument('--mode', choices=['full', 'minimal'], default='full',
                        help='Comparison mode: full (4 methods) or minimal (2 methods)')
    args = parser.parse_args()
    
    try:
        plot_h_cofgs_vs_f_s_comparison(mode=args.mode)
    except Exception as e:
        print(f"[ERROR] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)