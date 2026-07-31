#!/usr/bin/env python3
"""
DiatomScanNet 纲级别分类预测脚本
- 加载训练好的纲级别分类模型
- 对单张图像或批量图像进行预测
- 支持可视化结果
"""

import os
import sys
import json
import argparse
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import timm
import matplotlib.pyplot as plt
import seaborn as sns

# 字体设置

# Use unified prediction configuration
from diatom_cascade.config.prediction_config import PredictionConfig as Config
from diatom_cascade.runtime import load_checkpoint, load_label_encoder, get_preprocess_transforms
from diatom_cascade.models import FlatClassifier as EfficientNetClassifier

# Override checkpoint path for this model
Config.CHECKPOINT_PATH = Config.get_checkpoint_path("F-C")


def load_model(checkpoint_path):
    """加载训练好的模型"""
    print(f"正在加载模型: {checkpoint_path}")
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"模型文件不存在: {checkpoint_path}")
    
    # Use unified checkpoint loading
    checkpoint = load_checkpoint(checkpoint_path, Config.DEVICE)
    
    # 获取配置
    config = checkpoint['config']
    label_encoder = load_label_encoder(checkpoint, 'class_names')
    
    # 创建模型
    model = EfficientNetClassifier(
        num_classes=config['NUM_CLASSES'],
        model_name=config['MODEL_NAME'],
        pretrained=False,
    )
    
    # 加载权重
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(Config.DEVICE)
    model.eval()
    
    print(f"✅ 模型加载成功")
    print(f"类别数量: {config['NUM_CLASSES']}")
    print(f"类别名称: {list(label_encoder.classes_)}")
    
    return model, label_encoder, config

def preprocess_image(image_path, image_size=None):
    """预处理图像"""
    if image_size is None:
        image_size = Config.IMAGE_SIZE
    # 加载图像
    image = Image.open(image_path).convert('RGB')
    
    # Use unified transforms from config
    transform = get_preprocess_transforms()
    
    # 应用变换
    image_tensor = transform(image).unsqueeze(0)  # 添加batch维度
    
    return image_tensor, image

def predict_single_image(model, image_path, label_encoder, top_k=3):
    """预测单张图像"""
    print(f"\n正在预测: {image_path}")
    
    # 预处理
    image_tensor, original_image = preprocess_image(image_path)
    image_tensor = image_tensor.to(Config.DEVICE)
    
    # 预测
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        top_probs, top_indices = torch.topk(probabilities, top_k, dim=1)
    
    # 获取结果
    top_probs = top_probs.cpu().numpy()[0]
    top_indices = top_indices.cpu().numpy()[0]
    
    # 转换为类别名称
    top_classes = [label_encoder.classes_[idx] for idx in top_indices]
    
    # 打印结果
    print(f"前{top_k}个预测结果:")
    for i, (class_name, prob) in enumerate(zip(top_classes, top_probs)):
        print(f"  {i+1}. {class_name}: {prob:.4f} ({prob*100:.2f}%)")
    
    return {
        'image_path': image_path,
        'predictions': list(zip(top_classes, top_probs)),
        'original_image': original_image
    }

def predict_batch(model, image_dir, label_encoder, output_csv=None):
    """批量预测图像"""
    print(f"\n正在批量预测目录中的图片: {image_dir}")
    
    # 获取所有图像文件
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    image_files = []
    for ext in image_extensions:
        image_files.extend(Path(image_dir).glob(f"*{ext}"))
        image_files.extend(Path(image_dir).glob(f"*{ext.upper()}"))
    
    if not image_files:
        print("未找到图片文件!")
        return
    
    print(f"找到{len(image_files)}张图片")
    
    results = []
    
    for image_path in image_files:
        try:
            # 预处理
            image_tensor, _ = preprocess_image(image_path)
            image_tensor = image_tensor.to(Config.DEVICE)
            
            # 预测
            with torch.no_grad():
                outputs = model(image_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                top_prob, top_idx = torch.max(probabilities, dim=1)
            
            # 获取结果
            predicted_class = label_encoder.classes_[top_idx.item()]
            confidence = top_prob.item()
            
            results.append({
                'filename': image_path.name,
                'predicted_class': predicted_class,
                'confidence': confidence,
                'confidence_pct': confidence * 100
            })
            
            print(f"  {image_path.name}: {predicted_class} ({confidence:.4f})")
            
        except Exception as e:
            print(f"  处理错误 {image_path.name}: {e}")
            results.append({
                'filename': image_path.name,
                'predicted_class': 'ERROR',
                'confidence': 0.0,
                'confidence_pct': 0.0
            })
    
    # 保存结果
    if output_csv:
        df = pd.DataFrame(results)
        df.to_csv(output_csv, index=False)
        print(f"\n结果已保存到: {output_csv}")
    
    return results

def visualize_predictions(predictions, save_path=None):
    """可视化预测结果"""
    if not predictions:
        print("没有预测结果可可视化")
        return
    
    # 创建子图
    n_images = min(len(predictions), 9)  # 最多显示9张
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    axes = axes.flatten()
    
    for i, pred in enumerate(predictions[:n_images]):
        if i >= len(axes):
            break
            
        # 显示图像
        axes[i].imshow(pred['original_image'])
        axes[i].axis('off')
        
        # 添加预测结果
        top_class, top_prob = pred['predictions'][0]
        axes[i].set_title(f"{top_class}\n{top_prob:.3f}", fontsize=10, fontweight='semibold')
    
    # 隐藏多余的子图
    for i in range(n_images, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"可视化结果已保存到: {save_path}")
    
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='DiatomScanNet 纲级别分类预测')
    
    # Print separator for model identification
    print("\n" + "=" * 80)
    print("Predicting with Model: F-C")
    print("=" * 80 + "\n")
    
    parser.add_argument('--image', type=str, help='单张图像路径')
    parser.add_argument('--batch_dir', type=str, help='图像文件夹路径')
    parser.add_argument('--output_csv', type=str, help='批量预测结果CSV文件路径')
    parser.add_argument('--top_k', type=int, default=3, help='显示前K个预测结果')
    parser.add_argument('--visualize', action='store_true', help='显示可视化结果')
    parser.add_argument('--checkpoint', type=str, default=Config.CHECKPOINT_PATH, help='模型检查点路径')
    
    args = parser.parse_args()
    
    # 检查模型文件
    if not Path(args.checkpoint).exists():
        print(f"模型检查点未找到: {args.checkpoint}")
        print("请先训练模型或指定正确的检查点路径")
        return
    
    # 加载模型
    model, label_encoder, config = load_model(args.checkpoint)
    
    if args.image:
        # 单张图像预测
        if not Path(args.image).exists():
            print(f"图片未找到: {args.image}")
            return
        
        result = predict_single_image(model, args.image, label_encoder, args.top_k)
        
        if args.visualize:
            visualize_predictions([result])
    
    elif args.batch_dir:
        # 批量预测
        if not Path(args.batch_dir).exists():
            print(f"目录未找到: {args.batch_dir}")
            return
        
        results = predict_batch(model, args.batch_dir, label_encoder, args.output_csv)
        
        if args.visualize and results:
            # 为可视化准备数据
            viz_data = []
            for result in results[:9]:  # 最多9张
                if result['predicted_class'] != 'ERROR':
                    image_path = Path(args.batch_dir) / result['filename']
                    try:
                        _, original_image = preprocess_image(image_path)
                        viz_data.append({
                            'original_image': original_image,
                            'predictions': [(result['predicted_class'], result['confidence'])]
                        })
                    except:
                        continue
            
            if viz_data:
                visualize_predictions(viz_data)
    
    else:
        print("请指定 --image 或 --batch_dir")
        parser.print_help()

if __name__ == "__main__":
    main()
