#!/usr/bin/env python3
"""
Plot H-COF (Hierarchical Class + Order + Family) Training Curves
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


def plot_h_cof_training_curves(
    history: Union[dict, str, Path], 
    save_path: Union[str, Path] = None, 
    interactive: bool = False
):
    """Plot H-COF (Hierarchical Class + Order + Family) training curves."""
    plot_hierarchical_training_curves(
        history,
        levels=['class', 'order', 'family'],
        level_names=['Class', 'Order', 'Family'],
        save_path=save_path,
        interactive=interactive,
        ylabel_suffix='(Class + Order + Family)',
        model_title='H-COF: Hierarchical Model with Class, Order, and Family Levels',
        add_subtitles=True
    )


def main():
    """Command-line interface."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate H-COF training curves')
    parser.add_argument('--interactive', '-i', action='store_true',
                       help='Enable interactive mode')
    parser.add_argument('--history', type=str, default=None,
                       help='Path to history JSON file')
    parser.add_argument('--output', '-o', type=str, default=None,
                       help='Output path for plot')
    
    args = parser.parse_args()
    
    history_file = args.history or LOG_DIR / 'H_COF__training_history.json'
    output_file = args.output or REPORT_DIR / 'H_COF__training_curves.png'
    
    print(f"📊 Generating H-COF training curves...")
    print(f"   Input: {history_file}")
    print(f"   Output: {output_file}")
    
    if not Path(history_file).exists():
        print(f"   ⚠️  History file not found: {history_file}")
        return
    
    try:
        plot_h_cof_training_curves(
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