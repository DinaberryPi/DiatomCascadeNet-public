#!/usr/bin/env python3
"""
Progressive Comparison: F-C → H-CO → H-COF → H-COFG → H-COFGS
Compare level-wise argmax prediction across all levels
FIXED VERSION: Corrected title and figure size
"""

import sys
from pathlib import Path

from scripts.analysis.eval_figures._common import *
import argparse


def plot_progressive_methods_comparison(method='hierarchical'):
    """
    对比层层递进过程中，各推理方法在每个 level 上的表现变化。
    method 选项：
        - argmax: 逐层独立 argmax（无掩码）
        - hierarchical: Greedy hierarchical 预测（与论文主结果一致，默认）
        - beam: Beam search（beam width = 3）
    Models: F-C, H-CO, H-COF, H-COFG, H-COFGS
    """
    print(f"\n[INFO] Generating progressive comparison ({method})...")
    
    method_key_map = {
        'argmax': 'argmax_prediction',
        'hierarchical': 'hierarchical_prediction',
        'beam': 'beam_search_prediction'
    }
    
    if method not in method_key_map:
        raise ValueError(f"Unsupported method '{method}'. Choose from {list(method_key_map.keys())}.")
    
    eval_files = {
        'F-C': EVAL_DIR / 'F_C_evaluation_report.json',
        'H-CO': EVAL_DIR / 'H_CO_evaluation_report.json',
        'H-COF': EVAL_DIR / 'H_COF_evaluation_report.json',
        'H-COFG': EVAL_DIR / 'H_COFG_evaluation_report.json',
        'H-COFGS': EVAL_DIR / 'H_COFGS_evaluation_report.json'
    }
    
    models = ['F-C', 'H-CO', 'H-COF', 'H-COFG', 'H-COFGS']
    levels = ['Class', 'Order', 'Family', 'Genus', 'Species']
    
    # Load results
    results = {}
    for model, file_path in eval_files.items():
        if file_path.exists():
            with open(file_path, 'r') as f:
                results[model] = json.load(f)
        else:
            print(f"[WARN] {model} evaluation file not found: {file_path}")
    
    if not results:
        print("[ERROR] No evaluation results found!")
        return
    
    # Extract argmax metrics for each model at each level
    # Structure: metrics[level][model] = (accuracy, f1)
    level_metrics = {level: [] for level in levels}
    
    for model in models:
        if model not in results:
            for level in levels:
                level_metrics[level].append(None)
            continue
        
        data = results[model]
        
        # Select method block (fallback to root if missing, e.g., F-C / H-CO / H-COF)
        block_key = method_key_map[method]
        if block_key in data:
            method_data = data[block_key]
        else:
            method_data = data
        
        # Extract metrics based on model type
        if model == 'F-C':
            # F-C only has Class level, use overall_metrics
            acc = method_data.get('overall_metrics', {}).get('accuracy', None)
            f1 = method_data.get('overall_metrics', {}).get('f1_weighted', None)
            level_metrics['Class'].append((acc, f1))
            for level in ['Order', 'Family', 'Genus', 'Species']:
                level_metrics[level].append(None)
        
        elif model == 'H-CO':
            # H-CO has Class and Order
            class_acc = method_data.get('class', {}).get('accuracy', None)
            class_f1 = method_data.get('class', {}).get('f1_weighted', None)
            order_acc = method_data.get('order', {}).get('accuracy', None)
            order_f1 = method_data.get('order', {}).get('f1_weighted', None)
            
            level_metrics['Class'].append((class_acc, class_f1))
            level_metrics['Order'].append((order_acc, order_f1))
            for level in ['Family', 'Genus', 'Species']:
                level_metrics[level].append(None)
        
        elif model == 'H-COF':
            # H-COF has Class, Order, Family
            class_acc = method_data.get('class', {}).get('accuracy', None)
            class_f1 = method_data.get('class', {}).get('f1_weighted', None)
            order_acc = method_data.get('order', {}).get('accuracy', None)
            order_f1 = method_data.get('order', {}).get('f1_weighted', None)
            family_acc = method_data.get('family', {}).get('accuracy', None)
            family_f1 = method_data.get('family', {}).get('f1_weighted', None)
            
            level_metrics['Class'].append((class_acc, class_f1))
            level_metrics['Order'].append((order_acc, order_f1))
            level_metrics['Family'].append((family_acc, family_f1))
            for level in ['Genus', 'Species']:
                level_metrics[level].append(None)
        
        elif model == 'H-COFG':
            # H-COFG has Class, Order, Family, Genus
            class_acc = method_data.get('class_level', {}).get('accuracy', None)
            class_f1 = method_data.get('class_level', {}).get('f1_weighted', None)
            order_acc = method_data.get('order_level', {}).get('accuracy', None)
            order_f1 = method_data.get('order_level', {}).get('f1_weighted', None)
            family_acc = method_data.get('family_level', {}).get('accuracy', None)
            family_f1 = method_data.get('family_level', {}).get('f1_weighted', None)
            genus_acc = method_data.get('genus_level', {}).get('accuracy', None)
            genus_f1 = method_data.get('genus_level', {}).get('f1_weighted', None)
            
            level_metrics['Class'].append((class_acc, class_f1))
            level_metrics['Order'].append((order_acc, order_f1))
            level_metrics['Family'].append((family_acc, family_f1))
            level_metrics['Genus'].append((genus_acc, genus_f1))
            level_metrics['Species'].append(None)
        
        elif model == 'H-COFGS':
            # H-COFGS has all 5 levels
            class_acc = method_data.get('class_level', {}).get('accuracy', None)
            class_f1 = method_data.get('class_level', {}).get('f1_weighted', None)
            order_acc = method_data.get('order_level', {}).get('accuracy', None)
            order_f1 = method_data.get('order_level', {}).get('f1_weighted', None)
            family_acc = method_data.get('family_level', {}).get('accuracy', None)
            family_f1 = method_data.get('family_level', {}).get('f1_weighted', None)
            genus_acc = method_data.get('genus_level', {}).get('accuracy', None)
            genus_f1 = method_data.get('genus_level', {}).get('f1_weighted', None)
            species_acc = method_data.get('species_level', {}).get('accuracy', None)
            species_f1 = method_data.get('species_level', {}).get('f1_weighted', None)
            
            level_metrics['Class'].append((class_acc, class_f1))
            level_metrics['Order'].append((order_acc, order_f1))
            level_metrics['Family'].append((family_acc, family_f1))
            level_metrics['Genus'].append((genus_acc, genus_f1))
            level_metrics['Species'].append((species_acc, species_f1))
    
    # FIXED: Further narrowed figure size and adjusted height ratio
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10))
    
    x_positions = np.arange(len(models))
    # Updated colors to match dataset distribution colors for consistency
    colors_level = ['#EA8379', '#7DAEE0', '#B395BD', '#299D8F', '#E9C46A']  # Class, Order, Family, Genus, Species
    
    # Top: Accuracy Trend
    # First, collect all values at each position to detect overlaps
    position_values = {pos: [] for pos in x_positions}
    for i, level in enumerate(levels):
        values = []
        positions = []
        for j, metric in enumerate(level_metrics[level]):
            if metric is not None and metric[0] is not None:
                values.append(metric[0])
                positions.append(x_positions[j])
                position_values[x_positions[j]].append((metric[0], i, level))
        
        if values:
            ax1.plot(positions, values, marker='o', linewidth=3, markersize=10, 
                    label=level, color=colors_level[i], alpha=0.85)
    
    # Now add labels with intelligent offset to prevent ANY overlaps
    for i, level in enumerate(levels):
        values = []
        positions = []
        for j, metric in enumerate(level_metrics[level]):
            if metric is not None and metric[0] is not None:
                values.append(metric[0])
                positions.append(x_positions[j])
        
        if values:
            for idx, (pos, val) in enumerate(zip(positions, values)):
                # For the rightmost position (H-COFGS), put labels to the right
                if pos == positions[-1]:
                    # Place labels to the right of the point, staggered vertically by level
                    # Reverse order: Class at top (largest offset), Genus at bottom (smallest offset)
                    # For 4 levels at H-COFGS (Class, Order, Family, Genus), i = 0,1,2,3
                    num_levels_at_pos = len([v for v in position_values[pos] if v is not None])
                    offset_y = 0.025 - (i * 0.012)  # Reversed: higher index = lower position
                    # Dynamic horizontal offset based on x-axis range
                    x_offset = (len(models) - 1) * 0.04  # 4% of total x range
                    ax1.text(pos + x_offset, val + offset_y, f'{val:.3f}', 
                            ha='left', va='center',  # Left-aligned, centered vertically
                            fontweight='bold', fontsize=9,
                            color=colors_level[i])
                else:
                    # For other positions, use standard top offset
                    offset = 0.008
                    ax1.text(pos, val + offset, f'{val:.3f}', 
                            ha='center', va='bottom',
                            fontweight='bold', fontsize=9,
                            color=colors_level[i])
    
    title_map = {
        'argmax': 'Level-wise Argmax (No Constraints)',
        'hierarchical': 'Greedy Hierarchical',
        'beam': 'Beam Search (width=3)'
    }
    
    # Match image title format: "Accuracy (Greedy Hierarchical)"
    ax1.set_title(f'Accuracy ({title_map[method]})', 
                  fontsize=13, fontweight='semibold', pad=20)
    ax1.set_ylabel('Accuracy', fontsize=11, fontweight='semibold')
    ax1.set_xlabel('Model', fontsize=11, fontweight='semibold')
    ax1.set_xticks(x_positions)
    ax1.set_xticklabels(models, fontsize=10)
    
    # Dynamically set y-axis limits
    all_acc_values = []
    for level in levels:
        for metric in level_metrics[level]:
            if metric is not None and metric[0] is not None:
                all_acc_values.append(metric[0])
    if all_acc_values:
        y_min = max(0, min(all_acc_values) - 0.05)
        y_max = min(1.0, max(all_acc_values) + 0.05)  # Back to normal margin
        ax1.set_ylim(y_min, y_max)
    
    # Extend x-axis to make room for right-side labels
    x_margin = (len(models) - 1) * 0.15  # 15% of x range for right margin
    ax1.set_xlim(-0.2, len(models) - 1 + x_margin)
    
    ax1.legend(fontsize=9, loc='best', framealpha=0.9, ncol=5)
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(labelsize=9)
    
    # Bottom: F1 Trend
    # First, collect all values at each position to detect overlaps
    position_values_f1 = {pos: [] for pos in x_positions}
    for i, level in enumerate(levels):
        values = []
        positions = []
        for j, metric in enumerate(level_metrics[level]):
            if metric is not None and metric[1] is not None:
                values.append(metric[1])
                positions.append(x_positions[j])
                position_values_f1[x_positions[j]].append((metric[1], i, level))
        
        if values:
            ax2.plot(positions, values, marker='o', linewidth=3, markersize=10, 
                    label=level, color=colors_level[i], alpha=0.85)
    
    # Now add labels with intelligent offset to prevent ANY overlaps
    for i, level in enumerate(levels):
        values = []
        positions = []
        for j, metric in enumerate(level_metrics[level]):
            if metric is not None and metric[1] is not None:
                values.append(metric[1])
                positions.append(x_positions[j])
        
        if values:
            for idx, (pos, val) in enumerate(zip(positions, values)):
                # For the rightmost position (H-COFGS), put labels to the right
                if pos == positions[-1]:
                    # Place labels to the right of the point, staggered vertically by level
                    # Reverse order: Class at top (largest offset), Genus at bottom (smallest offset)
                    num_levels_at_pos = len([v for v in position_values_f1[pos] if v is not None])
                    offset_y = 0.025 - (i * 0.012)  # Reversed: higher index = lower position
                    # Dynamic horizontal offset based on x-axis range
                    x_offset = (len(models) - 1) * 0.04  # 4% of total x range
                    ax2.text(pos + x_offset, val + offset_y, f'{val:.3f}', 
                            ha='left', va='center',  # Left-aligned, centered vertically
                            fontweight='bold', fontsize=9,
                            color=colors_level[i])
                else:
                    # For other positions, use standard top offset
                    offset = 0.008
                    ax2.text(pos, val + offset, f'{val:.3f}', 
                            ha='center', va='bottom',
                            fontweight='bold', fontsize=9,
                            color=colors_level[i])
    
    # Match image title format: "Weighted F1 (Greedy Hierarchical)"
    ax2.set_title(f'Weighted F1 ({title_map[method]})', 
                  fontsize=13, fontweight='semibold', pad=20)
    ax2.set_ylabel('Weighted F1 Score', fontsize=11, fontweight='semibold')
    ax2.set_xlabel('Model', fontsize=11, fontweight='semibold')
    ax2.set_xticks(x_positions)
    ax2.set_xticklabels(models, fontsize=10)
    
    # Dynamically set y-axis limits
    all_f1_values = []
    for level in levels:
        for metric in level_metrics[level]:
            if metric is not None and metric[1] is not None:
                all_f1_values.append(metric[1])
    if all_f1_values:
        y_min = max(0, min(all_f1_values) - 0.05)
        y_max = min(1.0, max(all_f1_values) + 0.05)  # Back to normal margin
        ax2.set_ylim(y_min, y_max)
    
    # Extend x-axis to make room for right-side labels
    x_margin = (len(models) - 1) * 0.15  # 15% of x range for right margin
    ax2.set_xlim(-0.2, len(models) - 1 + x_margin)
    
    ax2.legend(fontsize=9, loc='best', framealpha=0.9, ncol=5)
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(labelsize=9)
    
    # Remove overall suptitle to match image - each subplot has its own title
    plt.tight_layout(rect=[0, 0, 1, 0.98], pad=2.0)
    
    output_name = 'progressive_comparison.png' if method == 'hierarchical' else f'progressive_comparison_{method}.png'
    plt.savefig(REPORT_DIR / output_name, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[INFO] Saved: {REPORT_DIR / output_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate progressive comparison across models')
    parser.add_argument('--method', choices=['argmax', 'hierarchical', 'beam'], default='hierarchical',
                        help='Prediction method to visualize (default: hierarchical)')
    args = parser.parse_args()
    
    try:
        plot_progressive_methods_comparison(method=args.method)
    except Exception as e:
        print(f"[ERROR] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)