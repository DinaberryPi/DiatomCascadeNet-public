#!/usr/bin/env python3
"""
Error Propagation Analysis: H-COFG vs F-G (REFERENCE ONLY - NOT FOR PAPER)
验证层级模型的"错误生物学合理性"假设 - Genus级别

NOTE: This is a reference script for completeness. 
The main error propagation analysis uses Species level (H-COFGS vs F-S).
This Genus-level analysis is not included in the paper as Species level is more comprehensive.
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import defaultdict
import torch
from sklearn.metrics import confusion_matrix
import sys

from diatom_cascade.config.path_config import get_data_root, get_output_dir
from diatom_cascade.runtime import load_checkpoint

# 设置
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

class ErrorPropagationAnalyzer:
    def __init__(self, taxonomy_tree_path, eval_results_dir):
        """
        参数:
            taxonomy_tree_path: 分类树JSON路径
            eval_results_dir: 评估结果目录
        """
        self.taxonomy_tree_path = Path(taxonomy_tree_path)
        self.eval_results_dir = Path(eval_results_dir)
        
        # 加载分类树
        with open(self.taxonomy_tree_path, 'r', encoding='utf-8') as f:
            taxonomy_data = json.load(f)
            self.taxonomy_tree = taxonomy_data.get('tree', {})
        
        # 构建查找映射
        self._build_lookup_maps()
    
    def _build_lookup_maps(self):
        """构建快速查找映射"""
        # Genus -> Family -> Order -> Class 映射
        self.genus_to_family = {}
        self.family_to_order = {}
        self.order_to_class = {}
        
        for class_name, class_data in self.taxonomy_tree.items():
            if not isinstance(class_data, dict):
                continue
            for order_name, order_data in class_data.items():
                if not isinstance(order_data, dict):
                    continue
                self.order_to_class[order_name] = class_name
                for family_name, family_data in order_data.items():
                    if not isinstance(family_data, dict):
                        continue
                    self.family_to_order[family_name] = order_name
                    for genus_name in family_data.keys():
                        self.genus_to_family[genus_name] = family_name
    
    def calculate_taxonomic_distance(self, true_genus, pred_genus):
        """
        计算两个genus之间的分类学距离
        
        返回:
            0: 完全正确
            1: 同Family，不同Genus
            2: 同Order，不同Family
            3: 同Class，不同Order
            4: 不同Class
        """
        if true_genus == pred_genus:
            return 0
        
        true_family = self.genus_to_family.get(true_genus)
        pred_family = self.genus_to_family.get(pred_genus)
        
        if true_family == pred_family and true_family is not None:
            return 1  # 同Family
        
        true_order = self.family_to_order.get(true_family) if true_family else None
        pred_order = self.family_to_order.get(pred_family) if pred_family else None
        
        if true_order == pred_order and true_order is not None:
            return 2  # 同Order
        
        true_class = self.order_to_class.get(true_order) if true_order else None
        pred_class = self.order_to_class.get(pred_order) if pred_order else None
        
        if true_class == pred_class and true_class is not None:
            return 3  # 同Class
        
        return 4  # 不同Class
    
    def load_encoders_from_checkpoint(self, checkpoint_path):
        """从checkpoint加载encoders"""
        checkpoint = load_checkpoint(checkpoint_path, "cpu")
        encoders = {}
        
        if 'class_names' in checkpoint:
            from sklearn.preprocessing import LabelEncoder
            class_encoder = LabelEncoder()
            class_encoder.fit(checkpoint['class_names'])
            encoders['class'] = class_encoder
        
        if 'order_names' in checkpoint:
            from sklearn.preprocessing import LabelEncoder
            order_encoder = LabelEncoder()
            order_encoder.fit(checkpoint['order_names'])
            encoders['order'] = order_encoder
        
        if 'family_names' in checkpoint:
            from sklearn.preprocessing import LabelEncoder
            family_encoder = LabelEncoder()
            family_encoder.fit(checkpoint['family_names'])
            encoders['family'] = family_encoder
        
        if 'genus_names' in checkpoint:
            from sklearn.preprocessing import LabelEncoder
            genus_encoder = LabelEncoder()
            genus_encoder.fit(checkpoint['genus_names'])
            encoders['genus'] = genus_encoder
        
        return encoders
    
    def convert_predictions_to_names(self, predictions, model_name, h_cofg_encoders=None):
        """转换预测结果ID为名称"""
        if predictions.get('format') == 'names':
            # 已经是名称格式
            return {
                'true_class': predictions['true_class'],
                'true_order': predictions['true_order'],
                'true_family': predictions['true_family'],
                'true_genus': predictions['true_genus'],
                'pred_class': predictions['pred_class'],
                'pred_order': predictions['pred_order'],
                'pred_family': predictions['pred_family'],
                'pred_genus': predictions['pred_genus']
            }
        
        # 需要转换ID到名称
        def id_to_name(ids, encoder):
            if isinstance(ids, list):
                return [encoder.classes_[id] if id < len(encoder.classes_) else 'UNKNOWN' for id in ids]
            return encoder.classes_[ids] if ids < len(encoder.classes_) else 'UNKNOWN'
        
        if model_name == 'H-COFG':
            if h_cofg_encoders is None:
                raise ValueError("H-COFG predictions need encoders to convert IDs to names")
            return {
                'true_class': id_to_name(predictions['true_class'], h_cofg_encoders.get('class')),
                'true_order': id_to_name(predictions['true_order'], h_cofg_encoders.get('order')),
                'true_family': id_to_name(predictions['true_family'], h_cofg_encoders.get('family')),
                'true_genus': id_to_name(predictions['true_genus'], h_cofg_encoders.get('genus')),
                'pred_class': id_to_name(predictions['pred_class'], h_cofg_encoders.get('class')),
                'pred_order': id_to_name(predictions['pred_order'], h_cofg_encoders.get('order')),
                'pred_family': id_to_name(predictions['pred_family'], h_cofg_encoders.get('family')),
                'pred_genus': id_to_name(predictions['pred_genus'], h_cofg_encoders.get('genus'))
            }
        else:
            # F-G: 已经是名称格式（从taxonomy lookup得到）
            return {
                'true_class': predictions['true_class'],
                'true_order': predictions['true_order'],
                'true_family': predictions['true_family'],
                'true_genus': predictions['true_genus'],
                'pred_class': predictions['pred_class'],
                'pred_order': predictions['pred_order'],
                'pred_family': predictions['pred_family'],
                'pred_genus': predictions['pred_genus']
            }
    
    def analyze_error_propagation(self, model_name, predictions_file, h_cofg_checkpoint_path=None):
        """
        分析单个模型的错误传播模式
        
        参数:
            model_name: 'H-COFG' 或 'F-G'
            predictions_file: 预测结果JSON文件路径
            h_cofg_checkpoint_path: H-COFG checkpoint路径（用于加载encoders）
        
        返回:
            dict: 错误分析结果
        """
        # 加载预测结果
        with open(predictions_file, 'r', encoding='utf-8') as f:
            predictions = json.load(f)
        
        # 转换预测结果为名称格式
        if model_name == 'H-COFG':
            if h_cofg_checkpoint_path is None:
                raise ValueError("H-COFG requires checkpoint path to load encoders")
            h_cofg_encoders = self.load_encoders_from_checkpoint(h_cofg_checkpoint_path)
            predictions = self.convert_predictions_to_names(predictions, model_name, h_cofg_encoders)
        else:
            predictions = self.convert_predictions_to_names(predictions, model_name)
        
        # 提取数据（现在都是名称）
        true_genus = np.array(predictions['true_genus'])
        pred_genus = np.array(predictions['pred_genus'])
        true_class = np.array(predictions['true_class'])
        true_order = np.array(predictions['true_order'])
        true_family = np.array(predictions['true_family'])
        pred_class = np.array(predictions['pred_class'])
        pred_order = np.array(predictions['pred_order'])
        pred_family = np.array(predictions['pred_family'])
        
        # 找出Genus错误的样本
        genus_errors = true_genus != pred_genus
        n_genus_errors = genus_errors.sum()
        
        if n_genus_errors == 0:
            print(f"{model_name}: No genus errors found!")
            return None
        
        # 计算错误样本的上层级正确率
        results = {
            'model': model_name,
            'total_samples': len(true_genus),
            'genus_errors': n_genus_errors,
            'genus_error_rate': n_genus_errors / len(true_genus),
            
            # 在Genus错误的样本中，上层级的正确率
            'family_correct_given_genus_error': (true_family[genus_errors] == pred_family[genus_errors]).sum() / n_genus_errors,
            'order_correct_given_genus_error': (true_order[genus_errors] == pred_order[genus_errors]).sum() / n_genus_errors,
            'class_correct_given_genus_error': (true_class[genus_errors] == pred_class[genus_errors]).sum() / n_genus_errors,
        }
        
        # 计算分类学距离分布
        distance_counts = defaultdict(int)
        total_distance = 0
        
        for i in range(len(true_genus)):
            if genus_errors[i]:
                dist = self.calculate_taxonomic_distance(true_genus[i], pred_genus[i])
                distance_counts[dist] += 1
                total_distance += dist
        
        results['taxonomic_distance_distribution'] = dict(distance_counts)
        results['mean_taxonomic_distance'] = total_distance / n_genus_errors if n_genus_errors > 0 else 0
        
        distance_labels = {
            0: 'Correct (0)',
            1: 'Same Family (1)',
            2: 'Same Order (2)',
            3: 'Same Class (3)',
            4: 'Different Class (4)'
        }
        results['distance_breakdown'] = {
            distance_labels[d]: count for d, count in distance_counts.items()
        }
        
        return results
    
    def compare_models(self, h_cofg_results, f_g_results):
        """对比两个模型的错误传播特性"""
        
        print("=" * 80)
        print("Error Propagation Analysis: H-COFG vs F-G (REFERENCE - NOT FOR PAPER)")
        print("=" * 80)
        
        print("\n1. Overall Statistics")
        print("-" * 80)
        
        for name, results in [('H-COFG Greedy', h_cofg_results), ('F-G Upper-level Lookup', f_g_results)]:
            print(f"\n{name}:")
            print(f"  Total samples: {results['total_samples']}")
            print(f"  Genus errors: {results['genus_errors']} ({results['genus_error_rate']:.2%})")
        
        print("\n2. Upper-level Correctness Given Genus Error")
        print("-" * 80)
        print("When Genus prediction is wrong, how often are upper levels correct?\n")
        
        levels = ['family', 'order', 'class']
        
        print(f"{'Level':<15} {'H-COFG':<15} {'F-G':<15} {'Difference':<15}")
        print("-" * 60)
        
        for level in levels:
            key = f'{level}_correct_given_genus_error'
            h_val = h_cofg_results[key]
            f_val = f_g_results[key]
            diff = h_val - f_val
            
            print(f"{level.capitalize():<15} {h_val:>6.2%}         {f_val:>6.2%}         {diff:>+6.2%}")
        
        print("\n3. Taxonomic Distance Distribution")
        print("-" * 80)
        print("How 'far' are the errors in taxonomic tree?\n")
        
        print(f"{'Distance':<30} {'H-COFG':<15} {'F-G':<15}")
        print("-" * 60)
        
        for dist in [1, 2, 3, 4]:
            h_dist = h_cofg_results['taxonomic_distance_distribution'].get(dist, 0)
            f_dist = f_g_results['taxonomic_distance_distribution'].get(dist, 0)
            
            h_pct = h_dist / h_cofg_results['genus_errors']
            f_pct = f_dist / f_g_results['genus_errors']
            
            labels = {
                1: 'Same Family (1)',
                2: 'Same Order (2)',
                3: 'Same Class (3)',
                4: 'Different Class (4)'
            }
            
            print(f"{labels[dist]:<30} {h_pct:>6.2%} ({h_dist:>3}) {f_pct:>6.2%} ({f_dist:>3})")
        
        print("\n4. Mean Taxonomic Distance")
        print("-" * 80)
        h_mean = h_cofg_results['mean_taxonomic_distance']
        f_mean = f_g_results['mean_taxonomic_distance']
        
        print(f"H-COFG: {h_mean:.3f}")
        print(f"F-G:    {f_mean:.3f}")
        print(f"Difference: {h_mean - f_mean:+.3f} ({'H-COFG worse' if h_mean > f_mean else 'H-COFG better'})")
        
        return {
            'H-COFG': h_cofg_results,
            'F-G': f_g_results
        }
    
    def plot_comparison(self, h_cofg_results, f_g_results, output_path):
        """可视化对比结果 - 2-Panel Figure"""
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
        
        # 定义颜色：粉色和蓝色
        color_h_cofg = '#FF6B9D'  # 粉色
        color_f_g = '#4A90E2'     # 蓝色
        
        # (A) Upper-level correctness given genus error
        ax1 = axes[0]
        levels = ['Family', 'Order', 'Class']
        h_vals = [
            h_cofg_results['family_correct_given_genus_error'],
            h_cofg_results['order_correct_given_genus_error'],
            h_cofg_results['class_correct_given_genus_error']
        ]
        f_vals = [
            f_g_results['family_correct_given_genus_error'],
            f_g_results['order_correct_given_genus_error'],
            f_g_results['class_correct_given_genus_error']
        ]
        
        x = np.arange(len(levels))
        width = 0.4  # 稍微减小宽度避免重叠
        
        bars1 = ax1.bar(x - width/2, h_vals, width, label='H-COFG Greedy Hierarchical Predict', 
                       color=color_h_cofg, alpha=0.85, edgecolor='white', linewidth=1.5)
        bars2 = ax1.bar(x + width/2, f_vals, width, label='F-G Upper-level Lookup', 
                       color=color_f_g, alpha=0.85, edgecolor='white', linewidth=1.5)
        
        ax1.set_xlabel('Taxonomic Level', fontsize=13, fontweight='medium')
        ax1.set_ylabel('Correctness Rate', fontsize=13, fontweight='medium')
        ax1.set_title('(A) Upper-Level Correctness Given Genus Error', fontsize=14, fontweight='bold', pad=12)
        ax1.set_xticks(x)
        ax1.set_xticklabels(levels, fontsize=12)
        ax1.set_ylim([0, 1.08])  # 增加一点空间避免标签被截断
        ax1.grid(axis='y', alpha=0.3, linestyle='--')
        
        # 添加数值标签（调整位置避免重叠）
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                # 如果值很小，标签放在柱子内部
                if height < 0.1:
                    va_pos = 'bottom'
                    y_offset = height + 0.01
                else:
                    va_pos = 'bottom'
                    y_offset = height + 0.02
                ax1.text(bar.get_x() + bar.get_width()/2., y_offset,
                        f'{height:.2%}',
                        ha='center', va=va_pos, fontsize=9.5, fontweight='medium')
        
        # (B) Taxonomic distance distribution
        ax2 = axes[1]
        
        distances = [1, 2, 3, 4]
        h_dist_counts = [h_cofg_results['taxonomic_distance_distribution'].get(d, 0) for d in distances]
        f_dist_counts = [f_g_results['taxonomic_distance_distribution'].get(d, 0) for d in distances]
        
        h_dist_pcts = [c / h_cofg_results['genus_errors'] for c in h_dist_counts]
        f_dist_pcts = [c / f_g_results['genus_errors'] for c in f_dist_counts]
        
        x = np.arange(len(distances))
        bars1 = ax2.bar(x - width/2, h_dist_pcts, width, label='H-COFG Greedy Hierarchical Predict', 
                       color=color_h_cofg, alpha=0.85, edgecolor='white', linewidth=1.5)
        bars2 = ax2.bar(x + width/2, f_dist_pcts, width, label='F-G Upper-level Lookup', 
                       color=color_f_g, alpha=0.85, edgecolor='white', linewidth=1.5)
        
        ax2.set_xlabel('Taxonomic Distance', fontsize=13, fontweight='medium')
        ax2.set_ylabel('Proportion of Errors', fontsize=13, fontweight='medium')
        ax2.set_title('(B) Distribution of Error Distances', fontsize=14, fontweight='bold', pad=12)
        ax2.set_xticks(x)
        ax2.set_xticklabels(['1\n(Same\nFamily)', '2\n(Same\nOrder)', 
                            '3\n(Same\nClass)', '4\n(Diff\nClass)'], fontsize=11)
        ax2.set_ylim([0, max(max(h_dist_pcts), max(f_dist_pcts)) * 1.15])  # 动态调整y轴范围
        ax2.grid(axis='y', alpha=0.3, linestyle='--')
        
        # 添加数值标签（只显示非零值，避免重叠）
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    # 如果值很小，标签放在柱子内部
                    if height < 0.05:
                        va_pos = 'bottom'
                        y_offset = height + 0.01
                    else:
                        va_pos = 'bottom'
                        y_offset = height + 0.02
                    ax2.text(bar.get_x() + bar.get_width()/2., y_offset,
                            f'{height:.1%}',
                            ha='center', va=va_pos, fontsize=9.5, fontweight='medium')
        
        # 共享legend（使用ax1的handles和labels，横向排列，放在中间）
        handles, labels = ax1.get_legend_handles_labels()
        fig.legend(handles, labels, loc='center', bbox_to_anchor=(0.5, -0.08), 
                  ncol=2, fontsize=11, framealpha=0.95, edgecolor='gray', fancybox=True)
        
        plt.tight_layout(pad=2.0)  # 增加padding避免重叠
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"\nSaved plot: {output_path}")
        
        return fig


def main():
    """主函数 - REFERENCE ONLY, NOT FOR PAPER"""
    
    print("\n" + "=" * 80)
    print("NOTE: This is a REFERENCE script for Genus-level error propagation analysis.")
    print("The main analysis uses Species level (H-COFGS vs F-S) and is included in the paper.")
    print("This Genus-level analysis is for reference only and NOT included in the paper.")
    print("=" * 80 + "\n")
    
    # 配置路径
    taxonomy_tree_path = get_data_root() / "preprocessed" / "taxonomy_tree.json"
    run_dir = get_output_dir()
    eval_results_dir = run_dir / "evaluation"
    output_dir = run_dir / "figures" / "error_propagation"
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Checkpoint路径（用于加载encoders）
    checkpoint_dir = run_dir / "checkpoints"
    h_cofg_checkpoint_path = checkpoint_dir / "best_H_COFG_model.pth"
    
    # 创建分析器
    analyzer = ErrorPropagationAnalyzer(taxonomy_tree_path, eval_results_dir)
    
    # 分析H-COFG (需要先运行评估脚本并保存预测结果)
    h_cofg_predictions = eval_results_dir / "H_COFG_greedy_predictions.json"
    f_g_predictions = eval_results_dir / "F_G_predictions.json"
    
    if not h_cofg_predictions.exists():
        print("WARNING: H-COFG prediction file not found.")
        print(f"Expected: {h_cofg_predictions}")
        print("Please run: python -m scripts.evaluate.evaluate_H_COFG")
        print("NOTE: You may need to modify evaluate_H_COFG.py to save predictions first.")
        return
    
    if not f_g_predictions.exists():
        print("WARNING: F-G prediction file not found.")
        print(f"Expected: {f_g_predictions}")
        print("Please run: python -m scripts.evaluate.evaluate_F_G")
        print("NOTE: You may need to modify evaluate_F_G.py to save predictions first.")
        return
    
    if not h_cofg_checkpoint_path.exists():
        print("WARNING: H-COFG checkpoint not found.")
        print(f"Expected: {h_cofg_checkpoint_path}")
        print("Cannot load encoders to convert IDs to names.")
        return
    
    print("Analyzing H-COFG...")
    h_cofg_results = analyzer.analyze_error_propagation(
        'H-COFG', 
        h_cofg_predictions,
        h_cofg_checkpoint_path=h_cofg_checkpoint_path
    )
    
    print("\nAnalyzing F-G...")
    f_g_results = analyzer.analyze_error_propagation('F-G', f_g_predictions)
    
    if h_cofg_results is None or f_g_results is None:
        print("WARNING: Analysis failed. Check if there are any errors in the predictions.")
        return
    
    # 对比分析
    print("\n")
    comparison = analyzer.compare_models(h_cofg_results, f_g_results)
    
    # 可视化
    output_path = output_dir / "error_propagation_H_COFG_vs_F_G_REFERENCE.png"
    analyzer.plot_comparison(h_cofg_results, f_g_results, output_path)
    
    # 保存结果（转换numpy类型为Python原生类型）
    def convert_to_serializable(obj):
        """递归转换numpy类型为Python原生类型"""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: convert_to_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        else:
            return obj
    
    results_json = output_dir / "error_propagation_H_COFG_vs_F_G_REFERENCE_results.json"
    comparison_serializable = convert_to_serializable(comparison)
    with open(results_json, 'w', encoding='utf-8') as f:
        json.dump(comparison_serializable, f, indent=2, ensure_ascii=False)
    print(f"Saved results: {results_json}")


if __name__ == "__main__":
    main()

