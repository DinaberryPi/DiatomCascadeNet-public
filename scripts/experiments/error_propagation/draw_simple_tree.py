# 用于生成简洁树形图的代码（Cursor可以运行）

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

from diatom_cascade.config.path_config import get_output_dir

def draw_taxonomy_tree_simple():
    """
    生成简洁的分类树，展示一个具体样本的错误对比
    """
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 12)
    ax.axis('off')
    
    # === 绘制分类树结构 ===
    # Level 0: Class
    class_nodes = {
        'Bacillariophyceae': (3, 10, 'lightgreen'),
        'Fragilariophyceae': (7, 10, 'lightcoral'),
    }
    
    for label, (x, y, color) in class_nodes.items():
        draw_node(ax, x, y, label, color=color, size='large')
    
    # Level 1: Order (under each class)
    # Bacillariophyceae → Naviculales, Eunotiales
    draw_node(ax, 1.5, 8, 'Naviculales', color='lightgreen')
    draw_node(ax, 4.5, 8, 'Eunotiales', color='lightyellow')  # H-COFGS wrong here
    
    # Fragilariophyceae → Fragilariales
    draw_node(ax, 7, 8, 'Fragilariales', color='lightcoral')
    
    # 连接线：Class → Order
    ax.plot([3, 1.5], [9.8, 8.2], 'g-', linewidth=2, label='Ground Truth')
    ax.plot([3, 4.5], [9.8, 8.2], 'orange', linestyle='--', linewidth=2, label='H-COFGS (distance=4)')
    ax.plot([7, 7], [9.8, 8.2], 'r:', linewidth=2, label='F-S (distance=5)')
    
    # Level 2: Family
    draw_node(ax, 1.5, 6, 'Naviculaceae', color='lightgreen')
    draw_node(ax, 4.5, 6, 'Eunotiaceae', color='lightyellow')
    draw_node(ax, 7, 6, 'Fragilariaceae', color='lightcoral')
    
    # 连接线
    ax.plot([1.5, 1.5], [7.8, 6.2], 'g-', linewidth=2)
    ax.plot([4.5, 4.5], [7.8, 6.2], 'orange', linestyle='--', linewidth=2)
    ax.plot([7, 7], [7.8, 6.2], 'r:', linewidth=2)
    
    # Level 3: Genus
    draw_node(ax, 1.5, 4, 'Navicula', color='lightgreen')
    draw_node(ax, 4.5, 4, 'Eunotia', color='lightyellow')
    draw_node(ax, 7, 4, 'Fragilaria', color='lightcoral')
    
    # 连接线
    ax.plot([1.5, 1.5], [5.8, 4.2], 'g-', linewidth=2)
    ax.plot([4.5, 4.5], [5.8, 4.2], 'orange', linestyle='--', linewidth=2)
    ax.plot([7, 7], [5.8, 4.2], 'r:', linewidth=2)
    
    # Level 4: Species (最底层)
    draw_node(ax, 1.5, 2, 'gallica ✓', color='lightgreen', size='small')
    draw_node(ax, 4.5, 2, 'bilunaris ✗', color='lightyellow', size='small')
    draw_node(ax, 7, 2, 'famelica ✗', color='lightcoral', size='small')
    
    # 连接线
    ax.plot([1.5, 1.5], [3.8, 2.2], 'g-', linewidth=2)
    ax.plot([4.5, 4.5], [3.8, 2.2], 'orange', linestyle='--', linewidth=2)
    ax.plot([7, 7], [3.8, 2.2], 'r:', linewidth=2)
    
    # === 添加标签和距离信息 ===
    ax.text(1.5, 1.2, 'Distance=0\n(Correct)', ha='center', fontsize=10, 
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
    ax.text(4.5, 1.2, 'Distance=4\n(Wrong Order)', ha='center', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7))
    ax.text(7, 1.2, 'Distance=5\n(Wrong Class)', ha='center', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.7))
    
    # === 图例 ===
    ax.text(0.5, 11.5, 'Sample: Navicula gallica', fontsize=14, fontweight='bold')
    ax.plot([], [], 'g-', linewidth=2, label='Ground Truth (Distance=0)')
    ax.plot([], [], color='orange', linestyle='--', linewidth=2, label='H-COFGS (Distance=4)')
    ax.plot([], [], 'r:', linewidth=2, label='F-S (Distance=5)')
    ax.legend(loc='upper right', fontsize=11)
    
    plt.tight_layout()
    output_path = get_output_dir() / 'figures' / 'error_propagation' / 'taxonomy_tree_simple.png'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ 简洁树图已生成: {output_path}")
    plt.close()

def draw_node(ax, x, y, label, color='gray', size='large'):
    """绘制分类树节点"""
    if size == 'large':
        width, height = 1.2, 0.6
        fontsize = 11
    else:
        width, height = 0.8, 0.4
        fontsize = 9
    
    box = FancyBboxPatch((x - width/2, y - height/2), width, height,
                          boxstyle="round,pad=0.05", 
                          edgecolor='black', facecolor=color, 
                          alpha=0.7, linewidth=1.5)
    ax.add_patch(box)
    ax.text(x, y, label, ha='center', va='center', fontsize=fontsize, fontweight='bold')

if __name__ == '__main__':
    draw_taxonomy_tree_simple()

