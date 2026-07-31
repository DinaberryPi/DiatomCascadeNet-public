#!/usr/bin/env python3
"""
Plot F-G (Family → Genus) Training Curves
"""
import json
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Union, List, Optional

# Import common utilities from same directory
from scripts.analysis.training_curves._common import (
    LOG_DIR, REPORT_DIR, 
    save_or_show_plot, plot_loss_subplot, 
    plot_accuracy_subplot, plot_f1_subplot, load_history
)


def plot_f_g_training_curves(
    history: Union[dict, str, Path], 
    save_path: Union[str, Path] = None, 
    interactive: bool = False,
    force_flat: bool = False
):
    """Plot F-G (Family → Genus) training curves.
    
    Auto-detects whether this is a flat genus-only model or hierarchical
    family→genus model based on the presence of meaningful family metrics.
    """
    history = load_history(history)
    
    # Detect flat vs hierarchical
    is_flat = force_flat
    if not is_flat:
        val_family_acc = history.get('val_family_acc', [])
        val_family_f1 = history.get('val_family_f1', [])
        train_family_acc = history.get('train_family_acc', [])
        
        has_meaningful_family = False
        if val_family_acc and max(val_family_acc) > 0.5:
            has_meaningful_family = True
        elif val_family_f1 and max(val_family_f1) > 0.5:
            has_meaningful_family = True
        elif train_family_acc and max(train_family_acc) > 0.5:
            has_meaningful_family = True
        
        is_flat = not has_meaningful_family
    
    if is_flat:
        _plot_flat_genus(history, save_path, interactive)
    else:
        _plot_hierarchical_family_genus(history, save_path, interactive)


def _plot_flat_genus(history: dict, save_path: Union[str, Path], interactive: bool):
    """Plot curves for flat genus-only model."""
    train_losses = history.get('train_losses', history.get('train_loss', []))
    val_losses = history.get('val_losses', history.get('val_loss', []))
    train_accs = history.get('train_genus_acc', history.get('train_accs', []))
    val_accs = history.get('val_genus_acc', history.get('val_accs', []))
    val_f1s = history.get('val_genus_f1', history.get('val_f1s', []))
    
    num_epochs = len(train_losses) if train_losses else len(val_losses)
    if num_epochs == 0:
        raise ValueError("No training data found in history")
    epochs = list(range(1, num_epochs + 1))
    
    # Convert accuracy to 0-1 range if needed
    if train_accs and train_accs[0] > 1:
        train_accs = [a / 100.0 for a in train_accs]
    if val_accs and val_accs[0] > 1:
        val_accs = [a / 100.0 for a in val_accs]
    
    # Create figure
    fig = plt.figure(figsize=(22, 8.5))
    
    total_height = 8.5
    gap_inches = 1.0
    title_y = 1.0 - (0.2 / total_height)
    top_margin = 1.0 - ((0.2 + gap_inches) / total_height)
    
    fig.suptitle('F-G: Flat Genus-Only Model Training Curves', 
                 fontsize=18, fontweight='bold', y=title_y)
    
    gs = fig.add_gridspec(1, 3, 
                          hspace=0.3, 
                          wspace=0.3,
                          left=0.06, right=0.98,
                          top=top_margin, bottom=0.08)
    
    # Loss
    ax_loss = fig.add_subplot(gs[0, 0])
    plot_loss_subplot(ax_loss, epochs, train_losses, val_losses, 
                     title='Training and Validation Loss',
                     subtitle='(Shows final epoch values)')
    
    # Accuracy
    ax_acc = fig.add_subplot(gs[0, 1])
    plot_accuracy_subplot(ax_acc, epochs, train_accs, val_accs, val_f1s,
                         title='Genus Accuracy',
                         subtitle='(Shows values at epoch with best F1 score)')
    
    # F1
    ax_f1 = fig.add_subplot(gs[0, 2])
    plot_f1_subplot(ax_f1, epochs, val_f1s, 
                   title='Genus Val Weighted F1',
                   subtitle='(Shows maximum value achieved)')
    
    save_or_show_plot(fig, save_path, interactive)


def _plot_hierarchical_family_genus(history: dict, save_path: Union[str, Path], interactive: bool):
    """Plot curves for hierarchical family→genus model."""
    train_losses = history.get('train_losses', history.get('train_loss', []))
    val_losses = history.get('val_losses', history.get('val_loss', []))
    
    train_family_accs = history.get('train_family_acc', [])
    val_family_accs = history.get('val_family_acc', [])
    val_family_f1s = history.get('val_family_f1', [])
    
    train_genus_accs = history.get('train_genus_acc', [])
    val_genus_accs = history.get('val_genus_acc', [])
    val_genus_f1s = history.get('val_genus_f1', [])
    
    num_epochs = len(train_losses) if train_losses else len(val_losses)
    if num_epochs == 0:
        raise ValueError("No training data found in history")
    epochs = list(range(1, num_epochs + 1))
    
    # Convert accuracy to 0-1 range if needed
    for accs_list in [train_family_accs, val_family_accs, train_genus_accs, val_genus_accs]:
        if accs_list and accs_list[0] > 1:
            accs_list[:] = [a / 100.0 for a in accs_list]
    
    # Create figure with 2 rows x 3 columns
    fig = plt.figure(figsize=(22, 14))
    
    total_height = 14
    gap_inches = 1.0
    title_y = 1.0 - (0.2 / total_height)
    top_margin = 1.0 - ((0.2 + gap_inches) / total_height)
    
    fig.suptitle('F-G: Hierarchical Family → Genus Model Training Curves', 
                 fontsize=18, fontweight='bold', y=title_y)
    
    gs = fig.add_gridspec(2, 3, 
                          hspace=0.4, 
                          wspace=0.3,
                          left=0.06, right=0.98,
                          top=top_margin, bottom=0.05)
    
    # Top row: Family
    ax_family_loss = fig.add_subplot(gs[0, 0])
    plot_loss_subplot(ax_family_loss, epochs, train_losses, val_losses, 
                     title='Training and Validation Loss',
                     subtitle='(Shows final epoch values)')
    
    ax_family_acc = fig.add_subplot(gs[0, 1])
    plot_accuracy_subplot(ax_family_acc, epochs, train_family_accs, val_family_accs, val_family_f1s,
                         title='Family Accuracy',
                         subtitle='(Shows values at epoch with best F1 score)')
    
    ax_family_f1 = fig.add_subplot(gs[0, 2])
    plot_f1_subplot(ax_family_f1, epochs, val_family_f1s, 
                   title='Family Val Weighted F1',
                   subtitle='(Shows maximum value achieved)')
    
    # Bottom row: Genus
    ax_genus_acc = fig.add_subplot(gs[1, 1])
    plot_accuracy_subplot(ax_genus_acc, epochs, train_genus_accs, val_genus_accs, val_genus_f1s,
                         title='Genus Accuracy',
                         subtitle='(Shows values at epoch with best F1 score)')
    
    ax_genus_f1 = fig.add_subplot(gs[1, 2])
    plot_f1_subplot(ax_genus_f1, epochs, val_genus_f1s, 
                   title='Genus Val Weighted F1',
                   subtitle='(Shows maximum value achieved)')
    
    save_or_show_plot(fig, save_path, interactive)


def main():
    """Command-line interface."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate F-G training curves')
    parser.add_argument('--interactive', '-i', action='store_true',
                       help='Enable interactive mode')
    parser.add_argument('--force-flat', action='store_true',
                       help='Force F-G to be treated as flat classifier')
    parser.add_argument('--history', type=str, default=None,
                       help='Path to history JSON file')
    parser.add_argument('--output', '-o', type=str, default=None,
                       help='Output path for plot')
    
    args = parser.parse_args()
    
    history_file = args.history or LOG_DIR / 'F_G__training_history.json'
    output_file = args.output or REPORT_DIR / 'F_G__training_curves.png'
    
    print(f"📊 Generating F-G training curves...")
    print(f"   Input: {history_file}")
    print(f"   Output: {output_file}")
    
    if not Path(history_file).exists():
        print(f"   ⚠️  History file not found: {history_file}")
        return
    
    try:
        plot_f_g_training_curves(
            history=str(history_file), 
            save_path=str(output_file), 
            interactive=args.interactive,
            force_flat=args.force_flat
        )
        print(f"   ✅ Saved: {output_file}")
    except Exception as e:
        print(f"   ❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()