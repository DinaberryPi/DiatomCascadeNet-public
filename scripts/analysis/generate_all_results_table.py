#!/usr/bin/env python3
"""
Generate comprehensive results table from all evaluation reports.
Reads JSON files directly to avoid hardcoding.
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional
from diatom_cascade.config.path_config import get_output_dir, get_project_root

# Project root
PROJECT_ROOT = get_project_root()
EVAL_DIR = get_output_dir(PROJECT_ROOT) / "evaluation"
OUTPUT_FILE = PROJECT_ROOT / "docs/all_results_tables.md"

# Model configurations
MODELS = {
    "F-C": {
        "file": "F_C_evaluation_report.json",
        "levels": ["Class"],
        "type": "flat"
    },
    "F-G": {
        "file": "F_G_evaluation_report.json",
        "levels": ["Genus"],
        "type": "flat"
    },
    "F-S": {
        "file": "F_S_evaluation_report.json",
        "levels": ["Class", "Order", "Family", "Genus", "Species"],
        "type": "flat"
    },
    "H-CO": {
        "file": "H_CO_evaluation_report.json",
        "levels": ["Class", "Order"],
        "type": "hierarchical"
    },
    "H-COF": {
        "file": "H_COF_evaluation_report.json",
        "levels": ["Class", "Order", "Family"],
        "type": "hierarchical"
    },
    "H-COFG": {
        "file": "H_COFG_evaluation_report.json",
        "levels": ["Class", "Order", "Family", "Genus"],
        "type": "hierarchical"
    },
    "H-COFGS": {
        "file": "H_COFGS_evaluation_report.json",
        "levels": ["Class", "Order", "Family", "Genus", "Species"],
        "type": "hierarchical"
    }
}

ALL_LEVELS = ["Class", "Order", "Family", "Genus", "Species"]


def load_evaluation_report(model_name: str) -> Optional[Dict[str, Any]]:
    """Load evaluation report JSON for a model."""
    config = MODELS.get(model_name)
    if not config:
        return None
    
    file_path = EVAL_DIR / config["file"]
    if not file_path.exists():
        print(f"Warning: {file_path} not found")
        return None
    
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def extract_metrics(data: Dict[str, Any], model_name: str, level: str) -> Dict[str, float]:
    """Extract accuracy and F1 for a specific level."""
    config = MODELS[model_name]
    metrics = {"accuracy": None, "f1_weighted": None}
    
    if config["type"] == "flat":
        if model_name == "F-C":
            # F-C only has overall_metrics
            if level == "Class" and "overall_metrics" in data:
                metrics["accuracy"] = data["overall_metrics"].get("accuracy")
                metrics["f1_weighted"] = data["overall_metrics"].get("f1_weighted")
        
        elif model_name == "F-G":
            # F-G structure similar to F-S but for Genus
            if level == "Genus" and "genus_metrics" in data:
                metrics["accuracy"] = data["genus_metrics"].get("accuracy")
                metrics["f1_weighted"] = data["genus_metrics"].get("f1_weighted")
            elif "upper_level_metrics" in data and level.lower() in data["upper_level_metrics"]:
                level_data = data["upper_level_metrics"][level.lower()]
                metrics["accuracy"] = level_data.get("accuracy")
                metrics["f1_weighted"] = level_data.get("f1_weighted")
        
        elif model_name == "F-S":
            if level == "Species" and "species_metrics" in data:
                metrics["accuracy"] = data["species_metrics"].get("accuracy")
                metrics["f1_weighted"] = data["species_metrics"].get("f1_weighted")
            elif "upper_level_metrics" in data and level.lower() in data["upper_level_metrics"]:
                level_data = data["upper_level_metrics"][level.lower()]
                metrics["accuracy"] = level_data.get("accuracy")
                metrics["f1_weighted"] = level_data.get("f1_weighted")
    
    elif config["type"] == "hierarchical":
        # Hierarchical models: use hierarchical_prediction (greedy)
        if model_name in ["H-CO", "H-COF"]:
            # Simple structure: class, order, family keys
            level_key = level.lower()
            if level_key in data:
                metrics["accuracy"] = data[level_key].get("accuracy")
                metrics["f1_weighted"] = data[level_key].get("f1_weighted")
        
        elif model_name in ["H-COFG", "H-COFGS"]:
            # Complex structure with hierarchical_prediction
            if "hierarchical_prediction" in data:
                hier_data = data["hierarchical_prediction"]
                level_key = f"{level.lower()}_level"
                if level_key in hier_data:
                    metrics["accuracy"] = hier_data[level_key].get("accuracy")
                    metrics["f1_weighted"] = hier_data[level_key].get("f1_weighted")
    
    return metrics


def format_value(value: Optional[float], decimals: int = 3) -> str:
    """Format a float value to exactly 3 decimal places or return '-' if None."""
    if value is None:
        return "-"
    # Always use 3 decimal places for consistency
    return f"{value:.3f}"


def generate_markdown_table() -> str:
    """Generate markdown table with selected results: H-COFGS vs F-S, Progressive Models, Error Propagation."""
    lines = []
    
    # Header
    lines.append("# Experimental Results Summary")
    lines.append("")
    lines.append("This document contains key experimental results.")
    lines.append("All values are automatically extracted from evaluation JSON files.")
    lines.append("")
    
    # Collect data
    all_data = {}
    for model_name in ["F-S", "H-COFGS", "F-C", "H-CO", "H-COF", "H-COFG"]:
        data = load_evaluation_report(model_name)
        if data:
            all_data[model_name] = data
    
    # 1. H-COFGS vs F-S Comparison
    lines.append("## H-COFGS vs F-S Comparison")
    lines.append("")
    
    # Use HTML table for proper merged cells
    lines.append("<table>")
    lines.append("<thead>")
    lines.append("  <tr>")
    lines.append("    <th rowspan=\"2\">Model</th>")
    for level in ALL_LEVELS:
        lines.append(f"    <th colspan=\"2\">{level}</th>")
    lines.append("  </tr>")
    lines.append("  <tr>")
    for _ in ALL_LEVELS:
        lines.append("    <th>Accuracy</th>")
        lines.append("    <th>Weighted F1</th>")
    lines.append("  </tr>")
    lines.append("</thead>")
    lines.append("<tbody>")
    
    # Rows for each model
    for model_name in ["H-COFGS", "F-S"]:
        if model_name not in all_data:
            continue
        
        lines.append("  <tr>")
        lines.append(f"    <td><strong>{model_name}</strong></td>")
        for level in ALL_LEVELS:
            metrics = extract_metrics(all_data[model_name], model_name, level)
            if metrics["accuracy"] is not None:
                lines.append(f"    <td>{format_value(metrics['accuracy'])}</td>")
                lines.append(f"    <td>{format_value(metrics['f1_weighted'])}</td>")
            else:
                lines.append("    <td>-</td>")
                lines.append("    <td>-</td>")
        lines.append("  </tr>")
    
    lines.append("</tbody>")
    lines.append("</table>")
    
    lines.append("")
    
    # 2. Progressive Models
    lines.append("## Progressive Models Performance")
    lines.append("")
    lines.append("Performance across progressive hierarchical models (Greedy Hierarchical Prediction).")
    lines.append("")
    
    # Use HTML table for proper merged cells
    lines.append("<table>")
    lines.append("<thead>")
    lines.append("  <tr>")
    lines.append("    <th rowspan=\"2\">Model</th>")
    for level in ALL_LEVELS:
        lines.append(f"    <th colspan=\"2\">{level}</th>")
    lines.append("  </tr>")
    lines.append("  <tr>")
    for _ in ALL_LEVELS:
        lines.append("    <th>Accuracy</th>")
        lines.append("    <th>Weighted F1</th>")
    lines.append("  </tr>")
    lines.append("</thead>")
    lines.append("<tbody>")
    
    # Rows for each model
    progressive_models = ["F-C", "H-CO", "H-COF", "H-COFG", "H-COFGS"]
    for model_name in progressive_models:
        if model_name not in all_data:
            continue
        
        lines.append("  <tr>")
        lines.append(f"    <td><strong>{model_name}</strong></td>")
        for level in ALL_LEVELS:
            config = MODELS.get(model_name, {})
            if level not in config.get("levels", []):
                lines.append("    <td>-</td>")
                lines.append("    <td>-</td>")
            else:
                metrics = extract_metrics(all_data[model_name], model_name, level)
                if metrics["accuracy"] is not None:
                    lines.append(f"    <td>{format_value(metrics['accuracy'])}</td>")
                    lines.append(f"    <td>{format_value(metrics['f1_weighted'])}</td>")
                else:
                    lines.append("    <td>-</td>")
                    lines.append("    <td>-</td>")
        lines.append("  </tr>")
    
    lines.append("</tbody>")
    lines.append("</table>")
    
    lines.append("")
    
    # 3. Error Propagation Results
    lines.append("## Error Propagation Analysis")
    lines.append("")
    lines.append("Error propagation analysis comparing H-COFGS and F-S models.")
    lines.append("")
    
    error_prop_file = PROJECT_ROOT / "report" / "error_propagation" / "error_propagation_H_COFGS_vs_F_S_results.json"
    if error_prop_file.exists():
        with open(error_prop_file, 'r', encoding='utf-8') as f:
            error_prop_data = json.load(f)
        
        h_cofgs_ep = error_prop_data.get("H-COFGS", {})
        f_s_ep = error_prop_data.get("F-S", {})
        
        if h_cofgs_ep and f_s_ep:
            lines.append("### Upper-Level Correctness Given Species Error")
            lines.append("")
            lines.append("| Level | H-COFGS | F-S |")
            lines.append("|------|---------|-----|")
            
            levels_ep = ["Genus", "Family", "Order", "Class"]
            for level in levels_ep:
                level_key = f"{level.lower()}_correct_given_species_error"
                h_val = h_cofgs_ep.get(level_key, 0)
                f_val = f_s_ep.get(level_key, 0)
                lines.append(f"| {level} | {format_value(h_val)} | {format_value(f_val)} |")
            
            lines.append("")
            lines.append("### Taxonomic Distance Distribution")
            lines.append("")
            lines.append("| Distance | H-COFGS | F-S |")
            lines.append("|----------|---------|-----|")
            
            h_dist = h_cofgs_ep.get("taxonomic_distance_distribution", {})
            f_dist = f_s_ep.get("taxonomic_distance_distribution", {})
            h_errors = h_cofgs_ep.get("species_errors", 1)
            f_errors = f_s_ep.get("species_errors", 1)
            
            distance_labels = {
                "1": "Same Genus",
                "2": "Same Family",
                "3": "Same Order",
                "4": "Same Class",
                "5": "Different Class"
            }
            
            for dist in ["1", "2", "3", "4", "5"]:
                h_count = h_dist.get(dist, 0)
                f_count = f_dist.get(dist, 0)
                h_pct = h_count / h_errors if h_errors > 0 else 0
                f_pct = f_count / f_errors if f_errors > 0 else 0
                label = distance_labels.get(dist, f"Distance {dist}")
                lines.append(f"| {label} | {format_value(h_pct)} | {format_value(f_pct)} |")
            
            lines.append("")
            lines.append(f"**Mean Taxonomic Distance:** H-COFGS: {format_value(h_cofgs_ep.get('mean_taxonomic_distance', 0))}, F-S: {format_value(f_s_ep.get('mean_taxonomic_distance', 0))}")
            lines.append("")
    else:
        lines.append("*Error propagation results not found. Run error propagation analysis first.*")
        lines.append("")
    
    lines.append("---")
    lines.append("")
    lines.append("*Generated automatically from evaluation JSON files.*")
    
    return "\n".join(lines)


def main():
    """Main function."""
    print("Generating all results table...")
    print(f"Evaluation directory: {EVAL_DIR}")
    print(f"Output file: {OUTPUT_FILE}")
    
    # Check if eval dir exists
    if not EVAL_DIR.exists():
        print(f"Error: Evaluation directory not found: {EVAL_DIR}")
        return
    
    # Generate markdown content
    try:
        content = generate_markdown_table()
    except Exception as e:
        print(f"Error generating table: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Write to file
    OUTPUT_FILE.parent.mkdir(exist_ok=True, parents=True)
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"✓ Generated: {OUTPUT_FILE}")
    processed = len([m for m in MODELS.keys() if (EVAL_DIR / MODELS[m]['file']).exists()])
    print(f"  Total models processed: {processed}/{len(MODELS)}")


if __name__ == "__main__":
    main()
