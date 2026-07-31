#!/usr/bin/env python3
"""
DiatomScanNet H-COF: Hierarchical Class + Order + Family Prediction
三层级联预测脚本: 纲 → 目 → 科
"""

import os
import sys
from pathlib import Path
import argparse
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import pandas as pd
import numpy as np
import timm
import matplotlib.pyplot as plt

from diatom_cascade.prediction import greedy_hierarchical_predict as hierarchical_predict
from diatom_cascade.config.prediction_config import PredictionConfig as Config
from diatom_cascade.models import ThreeLevelHierarchicalModel
from diatom_cascade.runtime import get_preprocess_transforms, load_checkpoint

# 字体设置


class HierarchicalPredictor:
    """三层级联预测器"""
    def __init__(self, model_path, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # 加载模型
        print(f"加载模型: {model_path}")
        checkpoint = load_checkpoint(model_path, self.device)
        
        self.num_classes = checkpoint['num_classes']
        self.num_orders = checkpoint['num_orders']
        self.num_families = checkpoint['num_families']
        self.class_names = checkpoint['class_names']
        self.order_names = checkpoint['order_names']
        self.family_names = checkpoint['family_names']
        self.M_class_order = checkpoint['M_class_order']
        self.M_order_family = checkpoint['M_order_family']
        
        self.model = ThreeLevelHierarchicalModel(
            self.num_classes,
            self.num_orders,
            self.num_families,
            Config.MODEL_NAME,
            pretrained=False,
        )
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # 图像预处理 - Use unified transforms
        self.transform = get_preprocess_transforms()
        
        print(f"✅ 模型加载成功")
        print(f"纲级别数量: {self.num_classes}")
        print(f"目级别数量: {self.num_orders}")
        print(f"科级别数量: {self.num_families}")
    
    def predict(self, image_path, topk=3):
        """预测单张图像"""
        # 加载图像
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # 预测
        with torch.no_grad():
            class_logits, order_logits, family_logits, class_probs, order_probs = self.model(image_tensor)
            
            # 层级预测 (使用统一的 greedy hierarchical_predict)
            pred_class, pred_order, pred_family = hierarchical_predict(
                class_logits, order_logits, 
                family_logits=family_logits,
                M_class_order=self.M_class_order, 
                M_order_family=self.M_order_family
            )
            
            # 转换为单个值（因为 batch_size=1）
            pred_class = pred_class[0].item()
            pred_order = pred_order[0].item()
            pred_family = pred_family[0].item()
            
            # 获取概率
            class_probs_np = torch.softmax(class_logits, dim=1).cpu().numpy()[0]
            order_probs_np = torch.softmax(order_logits, dim=1).cpu().numpy()[0]
            family_probs_np = torch.softmax(family_logits, dim=1).cpu().numpy()[0]
        
        # 整理结果
        result = {
            'class': {
                'name': self.class_names[pred_class],
                'probability': float(class_probs_np[pred_class]),
                'top_predictions': [
                    {
                        'name': self.class_names[i],
                        'probability': float(class_probs_np[i])
                    }
                    for i in np.argsort(class_probs_np)[::-1][:topk]
                ]
            },
            'order': {
                'name': self.order_names[pred_order],
                'probability': float(order_probs_np[pred_order]),
                'top_predictions': [
                    {
                        'name': self.order_names[i],
                        'probability': float(order_probs_np[i])
                    }
                    for i in np.argsort(order_probs_np)[::-1][:topk]
                ]
            },
            'family': {
                'name': self.family_names[pred_family],
                'probability': float(family_probs_np[pred_family]),
                'top_predictions': [
                    {
                        'name': self.family_names[i],
                        'probability': float(family_probs_np[i])
                    }
                    for i in np.argsort(family_probs_np)[::-1][:topk]
                ]
            }
        }
        
        return result
    
    def visualize(self, image_path, result, save_path=None):
        """可视化预测结果"""        
        image = Image.open(image_path).convert('RGB')
        
        fig, axes = plt.subplots(1, 4, figsize=(24, 6))
        
        # 显示图像
        axes[0].imshow(image)
        axes[0].axis('off')
        axes[0].set_title('输入图像', fontsize=14, fontweight='semibold')
        
        # 纲级别预测
        class_names = [p['name'] for p in result['class']['top_predictions']]
        class_probs = [p['probability'] for p in result['class']['top_predictions']]
        
        axes[1].barh(class_names, class_probs, color='steelblue')
        axes[1].set_xlabel('概率', fontsize=12)
        axes[1].set_title('纲级别预测', fontsize=14, fontweight='semibold')
        axes[1].set_xlim([0, 1])
        
        # 目级别预测
        order_names = [p['name'] for p in result['order']['top_predictions']]
        order_probs = [p['probability'] for p in result['order']['top_predictions']]
        
        axes[2].barh(order_names, order_probs, color='coral')
        axes[2].set_xlabel('概率', fontsize=12)
        axes[2].set_title('目级别预测', fontsize=14, fontweight='semibold')
        axes[2].set_xlim([0, 1])
        
        # 科级别预测
        family_names = [p['name'] for p in result['family']['top_predictions']]
        family_probs = [p['probability'] for p in result['family']['top_predictions']]
        
        axes[3].barh(family_names, family_probs, color='mediumseagreen')
        axes[3].set_xlabel('概率', fontsize=12)
        axes[3].set_title('科级别预测', fontsize=14, fontweight='semibold')
        axes[3].set_xlim([0, 1])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ 可视化结果已保存: {save_path}")
        else:
            plt.show()
        
        plt.close()

def main():
    parser = argparse.ArgumentParser(description='DiatomScanNet 三层级联预测 (纲→目→科)')
    
    # Print separator for model identification
    print("\n" + "=" * 80)
    print("Predicting with Model: H-COF")
    print("=" * 80 + "\n")
    
    parser.add_argument('--image', type=str, help='输入图像路径')
    parser.add_argument('--batch_dir', type=str, help='批量预测目录')
    parser.add_argument('--model', type=str, default='outputs/checkpoints/best_H_COF_model.pth', help='模型路径')
    parser.add_argument('--output_csv', type=str, help='输出CSV文件路径（批量预测时）')
    parser.add_argument('--visualize', action='store_true', help='是否可视化结果')
    parser.add_argument('--topk', type=int, default=3, help='显示前K个预测结果')
    
    args = parser.parse_args()
    
    # 创建预测器
    predictor = HierarchicalPredictor(args.model)
    
    if args.image:
        # 单张图像预测
        print(f"\n预测图像: {args.image}")
        result = predictor.predict(args.image, topk=args.topk)
        
        print("\n" + "=" * 60)
        print("预测结果")
        print("=" * 60)
        print(f"纲: {result['class']['name']} (概率: {result['class']['probability']:.4f})")
        print(f"目: {result['order']['name']} (概率: {result['order']['probability']:.4f})")
        print(f"科: {result['family']['name']} (概率: {result['family']['probability']:.4f})")
        print("=" * 60)
        
        print(f"\n纲级别Top-{args.topk}预测:")
        for i, pred in enumerate(result['class']['top_predictions'], 1):
            print(f"  {i}. {pred['name']}: {pred['probability']:.4f}")
        
        print(f"\n目级别Top-{args.topk}预测:")
        for i, pred in enumerate(result['order']['top_predictions'], 1):
            print(f"  {i}. {pred['name']}: {pred['probability']:.4f}")
        
        print(f"\n科级别Top-{args.topk}预测:")
        for i, pred in enumerate(result['family']['top_predictions'], 1):
            print(f"  {i}. {pred['name']}: {pred['probability']:.4f}")
        
        if args.visualize:
            save_path = Path(args.image).stem + '_hierarchical_prediction.png'
            predictor.visualize(args.image, result, save_path)
    
    elif args.batch_dir:
        # 批量预测
        print(f"\n批量预测目录: {args.batch_dir}")
        image_dir = Path(args.batch_dir)
        image_files = list(image_dir.glob('*.png')) + list(image_dir.glob('*.jpg'))
        
        print(f"找到 {len(image_files)} 张图像")
        
        results = []
        for img_path in image_files:
            print(f"预测: {img_path.name}")
            result = predictor.predict(str(img_path))
            
            results.append({
                'filename': img_path.name,
                'class': result['class']['name'],
                'class_probability': result['class']['probability'],
                'order': result['order']['name'],
                'order_probability': result['order']['probability'],
                'family': result['family']['name'],
                'family_probability': result['family']['probability']
            })
        
        # 保存结果
        df = pd.DataFrame(results)
        output_path = args.output_csv if args.output_csv else 'hierarchical_predictions.csv'
        df.to_csv(output_path, index=False)
        print(f"\n✅ 预测结果已保存: {output_path}")
    
    else:
        print("请指定 --image 或 --batch_dir")

if __name__ == "__main__":
    main()
