#!/usr/bin/env python3
"""
DiatomScanNet 纲到目分类预测脚本
使用自顶向下推理进行层级一致的预测
"""

import os
import sys
from pathlib import Path
import argparse
import json
import time
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import timm
from tqdm import tqdm
import matplotlib.pyplot as plt

# Add project root to path for utils
from diatom_cascade.prediction import greedy_hierarchical_predict as hierarchical_predict
from diatom_cascade.runtime import load_checkpoint, load_label_encoder
from diatom_cascade.models import ClassToOrderModel
from diatom_cascade.config.path_config import get_data_root

# 配置
class Config:
    # 数据路径
    DATA_ROOT = get_data_root()
    IMAGES_DIR = DATA_ROOT / "raw" / "images"
    
    # Use unified prediction configuration
    from diatom_cascade.config.prediction_config import PredictionConfig
    MODEL_NAME = PredictionConfig.MODEL_NAME
    IMAGE_SIZE = PredictionConfig.IMAGE_SIZE
    DEVICE = PredictionConfig.DEVICE
    
    # 输出路径
    OUTPUT_DIR = PredictionConfig.OUTPUT_DIR
    CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
    
    # 模型路径
    MODEL_PATH = CHECKPOINT_DIR / "best_H_CO_model.pth"


# topdown_predict is now replaced by hierarchical_predict from utils.hierarchical_predict

def load_model():
    """加载训练好的模型"""
    if not Config.MODEL_PATH.exists():
        raise FileNotFoundError(f"模型文件不存在: {Config.MODEL_PATH}")
    
    print(f"加载模型: {Config.MODEL_PATH}")
    checkpoint = load_checkpoint(Config.MODEL_PATH, 'cpu')
    
    # 重建编码器
    class_encoder = load_label_encoder(checkpoint, 'class_names')
    order_encoder = load_label_encoder(checkpoint, 'order_names')
    
    # 创建模型
    model = ClassToOrderModel(
        checkpoint['num_classes'], 
        checkpoint['num_orders'], 
        Config.MODEL_NAME,
        pretrained=False,
    )
    
    # 加载权重
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(Config.DEVICE)
    model.eval()
    
    # 加载掩膜矩阵
    M_class_order = checkpoint['M_class_order']
    
    print(f"✅ 模型加载成功")
    print(f"纲级别数量: {checkpoint['num_classes']}")
    print(f"目级别数量: {checkpoint['num_orders']}")
    print(f"验证准确率: {checkpoint['val_order_acc']:.4f}")
    
    return model, class_encoder, order_encoder, M_class_order

def predict_single_image(model, image_path, class_encoder, order_encoder, M_class_order, transform):
    """预测单张图像"""
    # 加载图像
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(Config.DEVICE)
    
    with torch.no_grad():
        class_logits, order_logits, class_probs = model(image_tensor)
        
        # 使用自顶向下预测
        pred_class, pred_order = hierarchical_predict(
            class_logits, order_logits, 
            M_class_order=M_class_order
        )
        
        # 获取预测结果
        pred_class_name = class_encoder.inverse_transform([pred_class.item()])[0]
        pred_order_name = order_encoder.inverse_transform([pred_order.item()])[0]
        
        # 获取概率
        class_prob = torch.softmax(class_logits, dim=1)[0, pred_class].item()
        order_prob = torch.softmax(order_logits, dim=1)[0, pred_order].item()
        
        # 获取top-k预测
        class_topk = torch.topk(torch.softmax(class_logits, dim=1), k=3)
        order_topk = torch.topk(torch.softmax(order_logits, dim=1), k=3)
        
        class_topk_names = [class_encoder.inverse_transform([i])[0] for i in class_topk.indices[0].cpu().numpy()]
        order_topk_names = [order_encoder.inverse_transform([i])[0] for i in order_topk.indices[0].cpu().numpy()]
        
        return {
            'pred_class': pred_class_name,
            'pred_order': pred_order_name,
            'class_prob': class_prob,
            'order_prob': order_prob,
            'class_topk': list(zip(class_topk_names, class_topk.values[0].cpu().numpy())),
            'order_topk': list(zip(order_topk_names, order_topk.values[0].cpu().numpy()))
        }

def predict_batch(model, image_paths, class_encoder, order_encoder, M_class_order, transform, batch_size=32):
    """批量预测"""
    results = []
    
    for i in tqdm(range(0, len(image_paths), batch_size), desc="预测中"):
        batch_paths = image_paths[i:i+batch_size]
        batch_images = []
        
        # 加载批次图像
        for path in batch_paths:
            image = Image.open(path).convert('RGB')
            image_tensor = transform(image)
            batch_images.append(image_tensor)
        
        batch_tensor = torch.stack(batch_images).to(Config.DEVICE)
        
        with torch.no_grad():
            class_logits, order_logits, class_probs = model(batch_tensor)
            
            # 使用自顶向下预测
            pred_classes, pred_orders = hierarchical_predict(
                class_logits, order_logits, 
                M_class_order=M_class_order
            )
            
            # 处理批次结果
            for j, path in enumerate(batch_paths):
                pred_class_name = class_encoder.inverse_transform([pred_classes[j].item()])[0]
                pred_order_name = order_encoder.inverse_transform([pred_orders[j].item()])[0]
                
                class_prob = torch.softmax(class_logits, dim=1)[j, pred_classes[j]].item()
                order_prob = torch.softmax(order_logits, dim=1)[j, pred_orders[j]].item()
                
                results.append({
                    'image_path': str(path),
                    'pred_class': pred_class_name,
                    'pred_order': pred_order_name,
                    'class_prob': class_prob,
                    'order_prob': order_prob
                })
    
    return results

def main():
    parser = argparse.ArgumentParser(description='DiatomScanNet 纲到目分类预测')
    
    # Print separator for model identification
    print("\n" + "=" * 80)
    print("Predicting with Model: H-CO")
    print("=" * 80 + "\n")
    
    parser.add_argument('--image', type=str, help='单张图像路径')
    parser.add_argument('--folder', type=str, help='图像文件夹路径')
    parser.add_argument('--output', type=str, default='predictions.json', help='输出文件路径')
    parser.add_argument('--batch_size', type=int, default=PredictionConfig.BATCH_SIZE, help='批次大小')
    
    args = parser.parse_args()
    
    print("\nDiatomScanNet H-CO: Hierarchical Class + Order Prediction")
    print(f"Device: {Config.DEVICE}")
    
    # Load model
    model, class_encoder, order_encoder, M_class_order = load_model()
    
    # Use unified transforms
    from diatom_cascade.runtime import get_preprocess_transforms
    transform = get_preprocess_transforms()
    
    # 预测
    if args.image:
        # 单张图像预测
        print(f"\n预测单张图像: {args.image}")
        result = predict_single_image(model, args.image, class_encoder, order_encoder, M_class_order, transform)
        
        print(f"\n预测结果:")
        print(f"纲: {result['pred_class']} (概率: {result['class_prob']:.4f})")
        print(f"目: {result['pred_order']} (概率: {result['order_prob']:.4f})")
        
        print(f"\n纲级别Top-3:")
        for name, prob in result['class_topk']:
            print(f"  {name}: {prob:.4f}")
        
        print(f"\n目级别Top-3:")
        for name, prob in result['order_topk']:
            print(f"  {name}: {prob:.4f}")
        
        # 保存结果
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"\n结果已保存到: {args.output}")
        
    elif args.folder:
        # 文件夹预测
        print(f"\n预测文件夹: {args.folder}")
        folder_path = Path(args.folder)
        
        # 获取所有图像文件
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        image_paths = [p for p in folder_path.iterdir() if p.suffix.lower() in image_extensions]
        
        if not image_paths:
            print("未找到图像文件")
            return
        
        print(f"找到 {len(image_paths)} 张图像")
        
        # 批量预测
        results = predict_batch(model, image_paths, class_encoder, order_encoder, M_class_order, transform, args.batch_size)
        
        # 保存结果
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        print(f"\n预测完成，结果已保存到: {args.output}")
        
        # 统计结果
        class_counts = defaultdict(int)
        order_counts = defaultdict(int)
        for result in results:
            class_counts[result['pred_class']] += 1
            order_counts[result['pred_order']] += 1
        
        print(f"\n预测统计:")
        print(f"纲级别分布:")
        for class_name, count in sorted(class_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"  {class_name}: {count}")
        
        print(f"\n目级别分布 (前10个):")
        for order_name, count in sorted(order_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
            print(f"  {order_name}: {count}")
    
    else:
        print("请指定 --image 或 --folder 参数")
        parser.print_help()

if __name__ == "__main__":
    main()
