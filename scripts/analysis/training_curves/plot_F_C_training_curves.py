#!/usr/bin/env python3
"""
Plot F-C (Flat Class-only) Training Curves
"""
import json
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Union

# Import common utilities from same directory
from scripts.analysis.training_curves._common import (
    LOG_DIR, REPORT_DIR, 
    save_or_show_plot, plot_loss_subplot, 
    plot_accuracy_subplot, plot_f1_subplot, load_history
)


def plot_f_c_training_curves(
    history: Union[dict, str, Path], 
    save_path: Union[str, Path] = None, 
    interactive: bool = False
):
    """Plot F-C (Flat Class-only) training curves."""
    history = load_history(history)
    
    train_losses = history.get('train_losses', history.get('train_loss', []))
    val_losses = history.get('val_losses', history.get('val_loss', []))
    train_accs = history.get('train_accs', history.get('train_acc', []))
    val_accs = history.get('val_accs', history.get('val_acc', []))
    val_f1s = history.get('val_f1s', history.get('val_f1', []))
    
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
    
    # Calculate spacing - need more gap for F-C due to 3-column layout
    total_height = 8.5
    gap_inches = 1.0  # Increased from 0.6 to give more space
    title_y = 1.0 - (0.2 / total_height)  # Move title slightly higher
    top_margin = 1.0 - ((0.2 + gap_inches) / total_height)
    
    fig.suptitle('F-C: Flat Class-Only Model Training Curves', 
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
                         title='Class Accuracy',
                         subtitle='(Shows values at epoch with best F1 score)')
    
    # F1
    ax_f1 = fig.add_subplot(gs[0, 2])
    plot_f1_subplot(ax_f1, epochs, val_f1s, 
                   title='Class Val Weighted F1',
                   subtitle='(Shows maximum value achieved)')
    
    save_or_show_plot(fig, save_path, interactive)


def main():
    """Command-line interface."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate F-C training curves')
    parser.add_argument('--interactive', '-i', action='store_true',
                       help='Enable interactive mode')
    parser.add_argument('--history', type=str, default=None,
                       help='Path to history JSON file')
    parser.add_argument('--output', '-o', type=str, default=None,
                       help='Output path for plot')
    
    args = parser.parse_args()
    
    history_file = args.history or LOG_DIR / 'F_C__training_history.json'
    output_file = args.output or REPORT_DIR / 'F_C__training_curves.png'
    
    print(f"📊 Generating F-C training curves...")
    print(f"   Input: {history_file}")
    print(f"   Output: {output_file}")
    
    if not Path(history_file).exists():
        print(f"   ⚠️  History file not found: {history_file}")
        return
    
    try:
        plot_f_c_training_curves(
            history=str(history_file), 
            save_path=str(output_file), 
            interactive=args.interactive
        )
        print(f"   ✅ Saved: {output_file}")
    except Exception as e:
        print(f"   ❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()