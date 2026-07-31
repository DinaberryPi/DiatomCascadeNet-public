#!/usr/bin/env python3
"""
Master script to run all model evaluations and generate paper figures
Run this after training all models

Models included:
- Flat baselines: F-C, F-G, F-S
- Hierarchical models: H-CO, H-COF, H-COFG, H-COFGS
"""

import subprocess
import sys
import argparse
import io
from pathlib import Path

from diatom_cascade.config.path_config import get_output_dir

# Fix encoding for Windows console
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

def run_command(cmd, description):
    """Run a command and report status"""
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError:
        print(f"Failed: {description}")
        raise

def main():
    parser = argparse.ArgumentParser(
        description='Run all model evaluations',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    args = parser.parse_args()
    
    # Check if models exist
    checkpoint_dir = get_output_dir() / "checkpoints"
    model_names = ["F-C", "F-G", "F-S", "H-CO", "H-COF", "H-COFG", "H-COFGS"]
    checkpoints = {
        checkpoint_dir / f"best_{name.replace('-', '_')}_model.pth": name
        for name in model_names
    }
    
    missing_models = []
    for checkpoint in checkpoints:
        if not checkpoint.exists():
            missing_models.append(checkpoint)
    
    if missing_models:
        print(f"\n{'='*80}")
        print(f"WARNING: {len(missing_models)} missing checkpoints detected")
        print(f"{'='*80}")
        print("\nMissing checkpoint files:")
        for i, checkpoint in enumerate(missing_models, 1):
            model_name = checkpoints[checkpoint]
            checkpoint_name = checkpoint.name
            print(f"  [{i}] {model_name}")
            print(f"      Checkpoint: {checkpoint_name}")
            print(f"      Full path: {checkpoint}")
        print(f"\n{'='*80}")
        print("To train the missing models, run the following commands:")
        print(f"{'='*80}")
        for checkpoint in missing_models:
            model_name = checkpoints[checkpoint]
            if model_name.startswith("F-") or model_name.startswith("H-"):
                train_script = f"scripts.train.train_{model_name.replace('-', '_')}"
            else:
                train_script = "scripts.train.train_<MODEL>"
            print(f"  python -m {train_script}")
        print(f"{'='*80}")
        raise FileNotFoundError("All seven checkpoints are required for the full evaluation run")
    
    evaluations = [
        ([sys.executable, "-m", "scripts.evaluate.evaluate_F_C"],
         "Evaluating F-C: Flat Class-only"),
        ([sys.executable, "-m", "scripts.evaluate.evaluate_F_G"],
         "Evaluating F-G: Flat Genus-only (for Error Propagation experiment)"),
        ([sys.executable, "-m", "scripts.evaluate.evaluate_F_S"],
         "Evaluating F-S: Flat Species-only"),
        ([sys.executable, "-m", "scripts.evaluate.evaluate_H_CO"],
         "Evaluating H-CO: Hierarchical Class + Order"),
        ([sys.executable, "-m", "scripts.evaluate.evaluate_H_COF"],
         "Evaluating H-COF: Hierarchical Class + Order + Family"),
        ([sys.executable, "-m", "scripts.evaluate.evaluate_H_COFG"],
         "Evaluating H-COFG: Hierarchical Class + Order + Family + Genus"),
        ([sys.executable, "-m", "scripts.evaluate.evaluate_H_COFGS"],
         "Evaluating H-COFGS: Hierarchical Class + Order + Family + Genus + Species")
    ]
    
    for cmd, desc in evaluations:
        run_command(cmd, desc)
    
    # Print final summary
    print("\n" + "=" * 80)
    print(f"All evaluations completed successfully: {len(evaluations)}/{len(evaluations)}")
    print("=" * 80)

if __name__ == "__main__":
    main()

