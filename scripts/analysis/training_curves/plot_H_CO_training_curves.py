#!/usr/bin/env python3
"""
Plot H-CO (Hierarchical Class + Order) Training Curves
"""
import json
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Union

# Import common utilities from same directory
from scripts.analysis.training_curves._common import (
    LOG_DIR, REPORT_DIR, 
    save_or_show_plot, plot_hierarchical_training_curves, load_history
)


def plot_h_co_training_curves(
    history: Union[dict, str, Path], 
    save_path: Union[str, Path] = None, 
    interactive: bool = False
):
    """Plot H-CO (Hierarchical Class + Order) training curves."""
    plot_hierarchical_training_curves(
        history,
        levels=['class', 'order'],
        level_names=['Class', 'Order'],
        save_path=save_path,
        interactive=interactive,
        ylabel_suffix='(Class + Order)',
        model_title='H-CO: Hierarchical Model with Class and Order Levels',
        add_subtitles=True  # 添加subtitle
    )


def main():
    """Command-line interface."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate H-CO training curves')
    parser.add_argument('--interactive', '-i', action='store_true',
                       help='Enable interactive mode')
    parser.add_argument('--history', type=str, default=None,
                       help='Path to history JSON file')
    parser.add_argument('--output', '-o', type=str, default=None,
                       help='Output path for plot')
    
    args = parser.parse_args()
    
    history_file = args.history or LOG_DIR / 'H_CO__training_history.json'
    output_file = args.output or REPORT_DIR / 'H_CO__training_curves.png'
    
    print(f"📊 Generating H-CO training curves...")
    print(f"   Input: {history_file}")
    print(f"   Output: {output_file}")
    
    if not Path(history_file).exists():
        print(f"   ⚠️  History file not found: {history_file}")
        return
    
    try:
        plot_h_co_training_curves(
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