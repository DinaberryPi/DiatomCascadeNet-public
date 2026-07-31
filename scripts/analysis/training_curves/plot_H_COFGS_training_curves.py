#!/usr/bin/env python3
"""
Plot H-COFGS (Class → Order → Family → Genus → Species) Training Curves
Generate training curve plots for the H-COFGS model
"""

from pathlib import Path
from scripts.analysis.training_curves._common import *
import argparse
import sys


def main():
    """Command-line interface."""
    parser = argparse.ArgumentParser(description='Generate H-COFGS training curves')
    parser.add_argument('--interactive', '-i', action='store_true',
                       help='Enable interactive mode')
    parser.add_argument('--history', type=str, default=None,
                       help='Path to history JSON file (default: auto-detect)')
    parser.add_argument('--output', type=str, default=None,
                       help='Output path (default: report/H_COFGS__training_curves.png)')
    
    args = parser.parse_args()
    
    history_file = Path(args.history) if args.history else LOG_DIR / 'H_COFGS__training_history.json'
    output_file = Path(args.output) if args.output else REPORT_DIR / 'H_COFGS__training_curves.png'
    
    print("📊 Generating H-COFGS training curves...")
    
    if not history_file.exists():
        print(f"❌ History file not found: {history_file}")
        sys.exit(1)
    
    try:
        plot_hierarchical_training_curves(
            str(history_file),
            levels=['class', 'order', 'family', 'genus', 'species'],
            level_names=['Class', 'Order', 'Family', 'Genus', 'Species'],
            save_path=str(output_file),
            interactive=args.interactive,
            ylabel_suffix='(Class + Order + Family + Genus + Species)',
            model_title='H-COFGS: Full Hierarchical Model with All Taxonomy Levels'
        )
        print(f"✅ Saved: {output_file}")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

