#!/usr/bin/env python3
"""
Error Propagation Analysis: H-COFGS vs F-S
验证层级模型的"错误生物学合理性"假设
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
import torch
from sklearn.metrics import confusion_matrix
import sys

from diatom_cascade.config.path_config import get_data_root, get_output_dir
from diatom_cascade.runtime import load_checkpoint

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
        # Species -> 所有上层级
        self.species_to_hierarchy = {}
        
        for class_name, class_data in self.taxonomy_tree.items():
            if not isinstance(class_data, dict):
                continue
            for order_name, order_data in class_data.items():
                if not isinstance(order_data, dict):
                    continue
                for family_name, family_data in order_data.items():
                    if not isinstance(family_data, dict):
                        continue
                    for genus_name, species_list in family_data.items():
                        if isinstance(species_list, list):
                            for species_name in species_list:
                                self.species_to_hierarchy[species_name] = {
                                    'class': class_name,
                                    'order': order_name,
                                    'family': family_name,
                                    'genus': genus_name,
                                    'species': species_name
                                }
        
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
    
    def calculate_taxonomic_distance(self, true_species, pred_species):
        """
        计算两个species之间的分类学距离
        
        返回:
            0: 完全正确
            1: 同Genus，不同Species
            2: 同Family，不同Genus
            3: 同Order，不同Family
            4: 同Class，不同Order
            5: 不同Class
        """
        if true_species == pred_species:
            return 0
        
        true_hier = self.species_to_hierarchy.get(true_species)
        pred_hier = self.species_to_hierarchy.get(pred_species)
        
        if not true_hier or not pred_hier:
            return 5  # 未知，视为最远距离
        
        # 从细到粗检查
        if true_hier['genus'] == pred_hier['genus']:
            return 1  # 同Genus
        if true_hier['family'] == pred_hier['family']:
            return 2  # 同Family
        if true_hier['order'] == pred_hier['order']:
            return 3  # 同Order
        if true_hier['class'] == pred_hier['class']:
            return 4  # 同Class
        return 5  # 不同Class
    
    def load_encoders_from_checkpoint(self, checkpoint_path):
        """从checkpoint加载encoders"""
        checkpoint = load_checkpoint(checkpoint_path, "cpu")
        
        encoders = {}
        
        # H-COFGS checkpoint格式
        if 'class_names' in checkpoint:
            from sklearn.preprocessing import LabelEncoder
            
            class_encoder = LabelEncoder()
            class_encoder.fit(checkpoint['class_names'])
            encoders['class'] = class_encoder
            
            order_encoder = LabelEncoder()
            order_encoder.fit(checkpoint['order_names'])
            encoders['order'] = order_encoder
            
            family_encoder = LabelEncoder()
            family_encoder.fit(checkpoint['family_names'])
            encoders['family'] = family_encoder
            
            genus_encoder = LabelEncoder()
            genus_encoder.fit(checkpoint['genus_names'])
            encoders['genus'] = genus_encoder
            
            species_encoder = LabelEncoder()
            species_encoder.fit(checkpoint['species_names'])
            encoders['species'] = species_encoder
        
        # F-S checkpoint格式
        elif 'label_encoder' in checkpoint:
            encoders['species'] = checkpoint['label_encoder']
            # F-S没有上层级encoder，需要从分类树构建
        
        return encoders
    
    def convert_predictions_to_names(self, predictions, model_name, h_cofgs_encoders=None):
        """将预测结果转换为species名称格式"""
        if predictions.get('format') == 'names':
            # F-S格式：已经是名称
            return {
                'true_class': predictions['true_class'],
                'true_order': predictions['true_order'],
                'true_family': predictions['true_family'],
                'true_genus': predictions['true_genus'],
                'true_species': predictions['true_species'],
                'pred_class': predictions['pred_class'],
                'pred_order': predictions['pred_order'],
                'pred_family': predictions['pred_family'],
                'pred_genus': predictions['pred_genus'],
                'pred_species': predictions['pred_species'],
            }
        else:
            # H-COFGS格式：ID需要转换为名称
            if h_cofgs_encoders is None:
                raise ValueError("H-COFGS predictions need encoders to convert IDs to names")
            
            def id_to_name(ids, encoder):
                if encoder is None:
                    return [str(id) for id in ids]
                return [encoder.classes_[id] if 0 <= id < len(encoder.classes_) else 'unknown' for id in ids]
            
            return {
                'true_class': id_to_name(predictions['true_class'], h_cofgs_encoders.get('class')),
                'true_order': id_to_name(predictions['true_order'], h_cofgs_encoders.get('order')),
                'true_family': id_to_name(predictions['true_family'], h_cofgs_encoders.get('family')),
                'true_genus': id_to_name(predictions['true_genus'], h_cofgs_encoders.get('genus')),
                'true_species': id_to_name(predictions['true_species'], h_cofgs_encoders.get('species')),
                'pred_class': id_to_name(predictions['pred_class'], h_cofgs_encoders.get('class')),
                'pred_order': id_to_name(predictions['pred_order'], h_cofgs_encoders.get('order')),
                'pred_family': id_to_name(predictions['pred_family'], h_cofgs_encoders.get('family')),
                'pred_genus': id_to_name(predictions['pred_genus'], h_cofgs_encoders.get('genus')),
                'pred_species': id_to_name(predictions['pred_species'], h_cofgs_encoders.get('species')),
            }
    
    def analyze_error_propagation(self, model_name, predictions_file, h_cofgs_checkpoint_path=None):
        """
        分析单个模型的错误传播模式
        
        参数:
            model_name: 'H-COFGS' 或 'F-S'
            predictions_file: 预测结果JSON文件路径
            h_cofgs_checkpoint_path: H-COFGS checkpoint路径（用于加载encoders）
        
        返回:
            dict: 错误分析结果
        """
        # 加载预测结果
        with open(predictions_file, 'r', encoding='utf-8') as f:
            predictions = json.load(f)
        
        # 转换预测结果为名称格式
        if model_name == 'H-COFGS':
            if h_cofgs_checkpoint_path is None:
                raise ValueError("H-COFGS requires checkpoint path to load encoders")
            h_cofgs_encoders = self.load_encoders_from_checkpoint(h_cofgs_checkpoint_path)
            predictions = self.convert_predictions_to_names(predictions, model_name, h_cofgs_encoders)
        else:
            predictions = self.convert_predictions_to_names(predictions, model_name)
        
        # 提取数据（现在都是名称）
        true_species = np.array(predictions['true_species'])
        pred_species = np.array(predictions['pred_species'])
        true_class = np.array(predictions['true_class'])
        true_order = np.array(predictions['true_order'])
        true_family = np.array(predictions['true_family'])
        true_genus = np.array(predictions['true_genus'])
        pred_class = np.array(predictions['pred_class'])
        pred_order = np.array(predictions['pred_order'])
        pred_family = np.array(predictions['pred_family'])
        pred_genus = np.array(predictions['pred_genus'])
        
        # 找出Species错误的样本
        species_errors = true_species != pred_species
        n_species_errors = species_errors.sum()
        
        if n_species_errors == 0:
            print(f"{model_name}: No species errors found!")
            return None
        
        # 计算错误样本的上层级正确率
        results = {
            'model': model_name,
            'total_samples': len(true_species),
            'species_errors': n_species_errors,
            'species_error_rate': n_species_errors / len(true_species),
            
            # 在Species错误的样本中，上层级的正确率
            'genus_correct_given_species_error': (true_genus[species_errors] == pred_genus[species_errors]).sum() / n_species_errors,
            'family_correct_given_species_error': (true_family[species_errors] == pred_family[species_errors]).sum() / n_species_errors,
            'order_correct_given_species_error': (true_order[species_errors] == pred_order[species_errors]).sum() / n_species_errors,
            'class_correct_given_species_error': (true_class[species_errors] == pred_class[species_errors]).sum() / n_species_errors,
        }
        
        # 计算分类学距离分布
        distances = []
        for t_sp, p_sp in zip(true_species[species_errors], pred_species[species_errors]):
            dist = self.calculate_taxonomic_distance(t_sp, p_sp)
            distances.append(dist)
        
        distance_counts = pd.Series(distances).value_counts().sort_index()
        results['taxonomic_distance_distribution'] = distance_counts.to_dict()
        
        # 平均距离
        results['mean_taxonomic_distance'] = np.mean(distances)
        
        # 详细：按距离分类错误
        distance_labels = {
            0: 'Correct (0)',
            1: 'Same Genus (1)',
            2: 'Same Family (2)',
            3: 'Same Order (3)',
            4: 'Same Class (4)',
            5: 'Different Class (5)'
        }
        results['distance_breakdown'] = {
            distance_labels[d]: count for d, count in distance_counts.items()
        }
        
        return results
    
    def compare_models(self, h_cofgs_results, f_s_results):
        """对比两个模型的错误传播特性"""
        
        print("=" * 80)
        print("Error Propagation Analysis: H-COFGS vs F-S")
        print("=" * 80)
        
        print("\n1. Overall Statistics")
        print("-" * 80)
        
        for name, results in [('H-COFGS Greedy', h_cofgs_results), ('F-S Upper-level Lookup', f_s_results)]:
            print(f"\n{name}:")
            print(f"  Total samples: {results['total_samples']}")
            print(f"  Species errors: {results['species_errors']} ({results['species_error_rate']:.2%})")
        
        print("\n2. Upper-level Correctness Given Species Error")
        print("-" * 80)
        print("When Species prediction is wrong, how often are upper levels correct?\n")
        
        levels = ['genus', 'family', 'order', 'class']
        
        print(f"{'Level':<15} {'H-COFGS':<15} {'F-S':<15} {'Difference':<15}")
        print("-" * 60)
        
        for level in levels:
            key = f'{level}_correct_given_species_error'
            h_val = h_cofgs_results[key]
            f_val = f_s_results[key]
            diff = h_val - f_val
            
            print(f"{level.capitalize():<15} {h_val:>6.2%}         {f_val:>6.2%}         {diff:>+6.2%}")
        
        print("\n3. Taxonomic Distance Distribution")
        print("-" * 80)
        print("How 'far' are the errors in taxonomic tree?\n")
        
        print(f"{'Distance':<30} {'H-COFGS':<15} {'F-S':<15}")
        print("-" * 60)
        
        for dist in [1, 2, 3, 4, 5]:
            h_dist = h_cofgs_results['taxonomic_distance_distribution'].get(dist, 0)
            f_dist = f_s_results['taxonomic_distance_distribution'].get(dist, 0)
            
            h_pct = h_dist / h_cofgs_results['species_errors']
            f_pct = f_dist / f_s_results['species_errors']
            
            labels = {
                1: 'Same Genus (1)',
                2: 'Same Family (2)',
                3: 'Same Order (3)',
                4: 'Same Class (4)',
                5: 'Different Class (5)'
            }
            
            print(f"{labels[dist]:<30} {h_pct:>6.2%} ({h_dist:>3}) {f_pct:>6.2%} ({f_dist:>3})")
        
        print("\n4. Mean Taxonomic Distance")
        print("-" * 80)
        h_mean = h_cofgs_results['mean_taxonomic_distance']
        f_mean = f_s_results['mean_taxonomic_distance']
        
        print(f"H-COFGS: {h_mean:.3f}")
        print(f"F-S:     {f_mean:.3f}")
        print(f"Difference: {h_mean - f_mean:+.3f} ({'H-COFGS worse' if h_mean > f_mean else 'H-COFGS better'})")
        
        return {
            'H-COFGS': h_cofgs_results,
            'F-S': f_s_results
        }
    


def main():
    """主函数"""
    
    # 配置路径
    taxonomy_tree_path = get_data_root() / "preprocessed" / "taxonomy_tree.json"
    run_dir = get_output_dir()
    eval_results_dir = run_dir / "evaluation"
    output_dir = run_dir / "figures" / "error_propagation"
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Checkpoint路径（用于加载encoders）
    checkpoint_dir = run_dir / "checkpoints"
    h_cofgs_checkpoint_path = checkpoint_dir / "best_H_COFGS_model.pth"
    
    # 创建分析器
    analyzer = ErrorPropagationAnalyzer(taxonomy_tree_path, eval_results_dir)
    
    # 分析H-COFGS (假设已经保存了预测结果)
    h_cofgs_predictions = eval_results_dir / "H_COFGS_greedy_predictions.json"
    f_s_predictions = eval_results_dir / "F_S_predictions.json"
    
    if not h_cofgs_predictions.exists():
        print("WARNING: H-COFGS prediction file not found.")
        print(f"Expected: {h_cofgs_predictions}")
        print("Please run: python -m scripts.evaluate.evaluate_H_COFGS")
        return
    
    if not f_s_predictions.exists():
        print("WARNING: F-S prediction file not found.")
        print(f"Expected: {f_s_predictions}")
        print("Please run: python -m scripts.evaluate.evaluate_F_S")
        return
    
    if not h_cofgs_checkpoint_path.exists():
        print("WARNING: H-COFGS checkpoint not found.")
        print(f"Expected: {h_cofgs_checkpoint_path}")
        print("Cannot load encoders to convert IDs to names.")
        return
    
    print("Analyzing H-COFGS...")
    h_cofgs_results = analyzer.analyze_error_propagation(
        'H-COFGS', 
        h_cofgs_predictions,
        h_cofgs_checkpoint_path=h_cofgs_checkpoint_path
    )
    
    print("\nAnalyzing F-S...")
    f_s_results = analyzer.analyze_error_propagation('F-S', f_s_predictions)
    
    if h_cofgs_results is None or f_s_results is None:
        print("WARNING: Analysis failed. Check if there are any errors in the predictions.")
        return
    
    # 对比分析
    print("\n")
    comparison = analyzer.compare_models(h_cofgs_results, f_s_results)
    
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
    
    results_json = output_dir / "error_propagation_H_COFGS_vs_F_S_results.json"
    comparison_serializable = convert_to_serializable(comparison)
    with open(results_json, 'w', encoding='utf-8') as f:
        json.dump(comparison_serializable, f, indent=2, ensure_ascii=False)
    print(f"Saved results: {results_json}")


if __name__ == "__main__":
    main()

