#!/usr/bin/env python3
"""
Common utilities for training curve plotting
Shared functions, paths, and helper utilities
"""

import json
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from typing import Dict, Union, Optional, List
from diatom_cascade.config.path_config import get_output_dir, get_project_root

PROJECT_ROOT = get_project_root()
LOG_DIR = get_output_dir(PROJECT_ROOT) / "logs"
REPORT_DIR = get_output_dir(PROJECT_ROOT) / "figures"

LOG_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)


def save_or_show_plot(fig, save_path: Optional[Union[str, Path]] = None, interactive: bool = False):
    """Helper function to save or show the plot with interactive mode support."""
    # Note: We don't use tight_layout here because we use gridspec with explicit layout parameters
    # The layout is already controlled by gridspec parameters (left, right, top, bottom, hspace, wspace)
    
    if interactive:
        plt.ion()
    
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        # bbox_inches='tight' will automatically adjust margins when saving
        plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0.2)
        if not interactive:
            plt.close()
        else:
            plt.show()
    else:
        plt.show()


def format_value(value: float) -> str:
    """Format value for display."""
    if abs(value) < 0.01:
        return f"{value:.4f}"
    elif abs(value) < 1:
        return f"{value:.4f}"
    else:
        return f"{value:.3f}"


def add_value_label(ax, x, y, label, color, offset_y=0.03):
    """Add a simple text label near the data point."""
    ylim = ax.get_ylim()
    y_range = ylim[1] - ylim[0]
    
    value_str = format_value(y)
    label_y = y + offset_y * y_range
    
    ax.text(x, label_y, f'{label}: {value_str}',
            fontsize=9, color=color, fontweight='bold',
            ha='center', va='bottom',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                     edgecolor=color, alpha=0.85, linewidth=1.2))


def plot_loss_subplot(ax, epochs, train_loss, val_loss, title="Loss", subtitle=""):
    """Plot loss subplot with final value labels."""
    if train_loss:
        ax.plot(epochs, train_loss, label='Train Loss', 
                linewidth=2.5, color='#2E8B57', marker='o', markersize=4, alpha=0.8)
    if val_loss:
        ax.plot(epochs, val_loss, label='Val Loss', 
                linewidth=2.5, color='#FF6347', marker='s', markersize=4, alpha=0.8)
    
    # Auto-adjust y-axis
    all_losses = (train_loss or []) + (val_loss or [])
    if all_losses:
        y_min, y_max = min(all_losses), max(all_losses)
        y_range = y_max - y_min
        ax.set_ylim([y_min - 0.15 * y_range, y_max + 0.15 * y_range])
    
    ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax.set_ylabel('Loss', fontsize=12, fontweight='bold')
    
    # Title with optional subtitle
    if subtitle:
        full_title = f'{title}\n{subtitle}'
        ax.set_title(full_title, fontsize=12, fontweight='bold', pad=20)
    else:
        ax.set_title(title, fontsize=13, fontweight='bold', pad=15)
    
    ax.legend(fontsize=11, loc='upper right', framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([epochs[0], epochs[-1]])
    
    # Add value labels at final epoch
    final_epoch = len(epochs)
    if train_loss and val_loss:
        # Determine which is higher to avoid overlap
        train_val = train_loss[-1]
        val_val = val_loss[-1]
        
        if train_val < val_val:
            # Train is lower, so put it below and Val above
            add_value_label(ax, final_epoch, train_val, 'Train', '#2E8B57', offset_y=-0.08)
            add_value_label(ax, final_epoch, val_val, 'Val', '#FF6347', offset_y=0.04)
        else:
            # Val is lower, so put it below and Train above
            add_value_label(ax, final_epoch, val_val, 'Val', '#FF6347', offset_y=-0.08)
            add_value_label(ax, final_epoch, train_val, 'Train', '#2E8B57', offset_y=0.04)
    elif train_loss:
        add_value_label(ax, final_epoch, train_loss[-1], 'Train', '#2E8B57', offset_y=0.04)
    elif val_loss:
        add_value_label(ax, final_epoch, val_loss[-1], 'Val', '#FF6347', offset_y=0.04)


def plot_accuracy_subplot(ax, epochs, train_acc, val_acc, val_f1, title="Accuracy", subtitle=""):
    """Plot accuracy subplot with labels at best F1 epoch."""
    if train_acc:
        ax.plot(epochs, train_acc, label='Train', 
                linewidth=2, color='#4169E1', marker='o', markersize=3, alpha=0.8)
    if val_acc:
        ax.plot(epochs, val_acc, label='Val', 
                linewidth=2, color='#FF1493', marker='s', markersize=3, alpha=0.8)
    
    ax.set_xlabel('Epoch', fontsize=11, fontweight='bold')
    ax.set_ylabel('Accuracy', fontsize=11, fontweight='bold')
    
    # Title with optional subtitle
    if subtitle:
        full_title = f'{title}\n{subtitle}'
        ax.set_title(full_title, fontsize=11, fontweight='bold', pad=20)
    else:
        ax.set_title(title, fontsize=12, fontweight='bold', pad=15)
    
    ax.legend(fontsize=10, loc='lower right', framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.10])
    ax.set_xlim([epochs[0], epochs[-1]])
    
    # Find epoch with best F1 score and add labels
    if val_f1 and len(val_f1) > 0:
        best_f1_epoch_idx = val_f1.index(max(val_f1))
        best_epoch = best_f1_epoch_idx + 1
        
        if train_acc and val_acc and best_f1_epoch_idx < len(train_acc) and best_f1_epoch_idx < len(val_acc):
            train_val = train_acc[best_f1_epoch_idx]
            val_val = val_acc[best_f1_epoch_idx]
            
            # Position labels to avoid overlap
            if train_val > val_val:
                add_value_label(ax, best_epoch, train_val, 'Train', '#4169E1', offset_y=0.03)
                add_value_label(ax, best_epoch, val_val, 'Val', '#FF1493', offset_y=-0.06)
            else:
                add_value_label(ax, best_epoch, val_val, 'Val', '#FF1493', offset_y=0.03)
                add_value_label(ax, best_epoch, train_val, 'Train', '#4169E1', offset_y=-0.06)
        elif train_acc and best_f1_epoch_idx < len(train_acc):
            add_value_label(ax, best_epoch, train_acc[best_f1_epoch_idx], 'Train', '#4169E1', offset_y=0.03)
        elif val_acc and best_f1_epoch_idx < len(val_acc):
            add_value_label(ax, best_epoch, val_acc[best_f1_epoch_idx], 'Val', '#FF1493', offset_y=0.03)


def plot_f1_subplot(ax, epochs, val_f1, title="Val Weighted F1", subtitle=""):
    """Plot F1 subplot with max value label."""
    if not val_f1:
        ax.text(0.5, 0.5, 'F1 data\nnot available', ha='center', va='center', 
                transform=ax.transAxes, fontsize=11, color='gray')
        ax.set_xlabel('Epoch', fontsize=11, fontweight='bold')
        ax.set_ylabel('Weighted F1', fontsize=11, fontweight='bold')
        # Title with optional subtitle for empty case
        if subtitle:
            full_title = f'{title}\n{subtitle}'
            ax.set_title(full_title, fontsize=11, fontweight='bold', pad=20)
        else:
            ax.set_title(title, fontsize=12, fontweight='bold')
        return
    
    ax.plot(epochs, val_f1, label='Val Weighted F1', 
            linewidth=2, color='#9370DB', marker='o', markersize=3, alpha=0.8)
    
    ax.set_xlabel('Epoch', fontsize=11, fontweight='bold')
    ax.set_ylabel('Weighted F1', fontsize=11, fontweight='bold')
    
    # Title with optional subtitle
    if subtitle:
        full_title = f'{title}\n{subtitle}'
        ax.set_title(full_title, fontsize=11, fontweight='bold', pad=20)
    else:
        ax.set_title(title, fontsize=12, fontweight='bold', pad=15)
    
    ax.legend(fontsize=10, loc='lower right', framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.10])
    ax.set_xlim([epochs[0], epochs[-1]])
    
    # Add label at max F1 value
    max_f1_idx = val_f1.index(max(val_f1))
    max_epoch = max_f1_idx + 1
    add_value_label(ax, max_epoch, val_f1[max_f1_idx], 'Max', '#9370DB', offset_y=0.03)


def load_history(history: Union[Dict, str, Path]) -> Dict:
    """Load history from file or return dict."""
    if isinstance(history, (str, Path)):
        with open(history, 'r') as f:
            return json.load(f)
    return history


def plot_hierarchical_training_curves(
    history: Union[Dict, str, Path],
    levels: List[str],
    level_names: List[str],
    save_path: Union[str, Path] = None,
    interactive: bool = False,
    ylabel_suffix: str = "",
    model_title: str = "",
    add_subtitles: bool = False
):
    """Generic function to plot hierarchical training curves."""
    history = load_history(history)
    
    train_loss = history.get('train_loss', history.get('train_losses', []))
    val_loss = history.get('val_loss', history.get('val_losses', []))
    
    num_epochs = len(train_loss) if train_loss else len(val_loss) if val_loss else 0
    if num_epochs == 0:
        raise ValueError("No training data found in history")
    epochs = list(range(1, num_epochs + 1))
    
    # Extract metrics for each level
    metrics = {}
    for level in levels:
        metrics[level] = {
            'train_acc': history.get(f'train_{level}_acc', history.get(f'train_{level}_accs', [])),
            'val_acc': history.get(f'val_{level}_acc', history.get(f'val_{level}_accs', [])),
            'train_loss': history.get(f'train_{level}_loss', []),
            'val_loss': history.get(f'val_{level}_loss', []),
            'val_f1': history.get(f'val_{level}_f1', history.get(f'val_{level}_f1s', []))
        }
    
    # Create figure with proper spacing
    num_rows = len(levels) + 1
    fig = plt.figure(figsize=(24, 5.2 * num_rows + 1.0))
    
    # 调整title到图的距离
    total_height = 5.2 * num_rows + 1.0
    gap_inches = 1.2
    title_y = 1.0 - (0.3 / total_height)
    top_margin = 1.0 - ((0.3 + gap_inches) / total_height)
    
    fig.suptitle(model_title, fontsize=18, fontweight='bold', y=title_y)
    
    gs = fig.add_gridspec(num_rows, 3, 
                          hspace=0.4,  # 减少从0.7到0.4
                          wspace=0.3,
                          left=0.06, right=0.98,
                          top=top_margin, bottom=0.03)
    
    # TOP ROW: Combined Loss
    ax_loss = fig.add_subplot(gs[0, :])
    loss_subtitle = "(Shows final epoch values)" if add_subtitles else ""
    plot_loss_subplot(
        ax_loss, epochs, train_loss, val_loss,
        title=f'Training and Validation Loss {ylabel_suffix}',
        subtitle=loss_subtitle
    )
    
    # Level-specific rows
    for row_idx, (level, level_name) in enumerate(zip(levels, level_names), start=1):
        m = metrics[level]
        
        # Accuracy
        ax_acc = fig.add_subplot(gs[row_idx, 0])
        acc_subtitle = "(Shows values at epoch with best F1 score)" if add_subtitles else ""
        plot_accuracy_subplot(
            ax_acc, epochs, m['train_acc'], m['val_acc'], m['val_f1'],
            title=f'{level_name} Accuracy',
            subtitle=acc_subtitle
        )
        
        # Loss
        ax_loss_level = fig.add_subplot(gs[row_idx, 1])
        if m['train_loss'] or m['val_loss']:
            loss_subtitle = "(Shows final epoch values)" if add_subtitles else ""
            plot_loss_subplot(
                ax_loss_level, epochs, m['train_loss'], m['val_loss'],
                title=f'{level_name} Loss',
                subtitle=loss_subtitle
            )
        else:
            ax_loss_level.text(0.5, 0.5, 'Loss data\nnot available', 
                              ha='center', va='center', transform=ax_loss_level.transAxes, 
                              fontsize=11, color='gray')
            ax_loss_level.set_xlabel('Epoch', fontsize=11, fontweight='bold')
            ax_loss_level.set_ylabel('Loss', fontsize=11, fontweight='bold')
            ax_loss_level.set_title(f'{level_name} Loss', fontsize=12, fontweight='bold')
        
        # F1
        ax_f1 = fig.add_subplot(gs[row_idx, 2])
        f1_subtitle = "(Shows maximum value achieved)" if add_subtitles else ""
        plot_f1_subplot(ax_f1, epochs, m['val_f1'], 
                       title=f'{level_name} Val Weighted F1',
                       subtitle=f1_subtitle)
    
    save_or_show_plot(fig, save_path, interactive)
