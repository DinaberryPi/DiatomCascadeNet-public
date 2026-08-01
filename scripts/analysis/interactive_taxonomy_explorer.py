"""
Notebook-friendly Taxonomy Visualization
Usage in Jupyter:
    from analysis.interactive_taxonomy_explorer import generate_statistics_png
    generate_statistics_png()
"""

import json
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import sys
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

from diatom_cascade.config.path_config import get_data_root, get_output_dir


REPORT_DIR = get_output_dir() / "figures"

# Set UTF-8 encoding for output
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

def load_taxonomy_and_stats():
    """Load taxonomy tree and statistics"""
    data_root = get_data_root()
    TAXONOMY_JSON = data_root / "preprocessed" / "taxonomy_tree.json"
    LABELS_CSV = data_root / "cleaned" / "labels_clean.csv"
    
    with open(TAXONOMY_JSON, 'r', encoding='utf-8') as f:
        taxonomy = json.load(f)
    
    df = pd.read_csv(LABELS_CSV)
    
    return taxonomy['tree'], taxonomy['statistics'], df

def create_statistics_dashboard(stats, df):
    """Create statistics dashboard"""
    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=('Class Distribution', 'Top 10 Orders', 
                       'Top 10 Families', 'Top 15 Genera',
                       'Top 15 Species', ''),
        specs=[[{"type": "bar"}, {"type": "bar"}],
               [{"type": "bar"}, {"type": "bar"}],
               [{"type": "bar"}, {"type": "bar"}]],
        vertical_spacing=0.12,
        horizontal_spacing=0.15
    )
    
    # Class distribution
    class_counts = pd.Series(stats['class_distribution'])
    fig.add_trace(
        go.Bar(x=class_counts.index, y=class_counts.values, 
               marker_color='#ff7f0e', name='Classes',
               text=class_counts.values, textposition='outside'),
        row=1, col=1
    )
    
    # Top 10 Orders
    order_counts = pd.Series(stats['order_distribution']).sort_values(ascending=False).head(10)
    fig.add_trace(
        go.Bar(x=order_counts.values, y=order_counts.index, 
               orientation='h', marker_color='#2ca02c', name='Orders',
               text=order_counts.values, textposition='outside'),
        row=1, col=2
    )
    
    # Top 10 Families
    family_counts = pd.Series(stats['family_distribution']).sort_values(ascending=False).head(10)
    fig.add_trace(
        go.Bar(x=family_counts.values, y=family_counts.index, 
               orientation='h', marker_color='#d62728', name='Families',
               text=family_counts.values, textposition='outside'),
        row=2, col=1
    )
    
    # Top 15 Genera
    genus_counts = pd.Series(stats['genus_distribution']).sort_values(ascending=False).head(15)
    fig.add_trace(
        go.Bar(x=genus_counts.values, y=genus_counts.index, 
               orientation='h', marker_color='#9467bd', name='Genera',
               text=genus_counts.values, textposition='outside'),
        row=2, col=2
    )
    
    # Top 15 Species
    if 'species_distribution' in stats:
        species_counts = pd.Series(stats['species_distribution']).sort_values(ascending=False).head(15)
    else:
        species_counts = df['species'].value_counts().sort_values(ascending=False).head(15)
    fig.add_trace(
        go.Bar(x=species_counts.values, y=species_counts.index, 
               orientation='h', marker_color='#8c564b', name='Species',
               text=species_counts.values, textposition='outside'),
        row=3, col=1
    )
    
    # Update axes
    fig.update_xaxes(title_text="Sample Count", row=1, col=1)
    fig.update_xaxes(title_text="Sample Count", row=2, col=1)
    fig.update_xaxes(title_text="Sample Count", row=3, col=1)
    fig.update_xaxes(title_text="Sample Count", row=1, col=2)
    fig.update_xaxes(title_text="Sample Count", row=2, col=2)
    fig.update_xaxes(visible=False, row=3, col=2)
    
    fig.update_yaxes(title_text="Class", row=1, col=1)
    fig.update_yaxes(title_text="Order", row=1, col=2)
    fig.update_yaxes(title_text="Family", row=2, col=1)
    fig.update_yaxes(title_text="Genus", row=2, col=2)
    fig.update_yaxes(title_text="Species", row=3, col=1)
    fig.update_yaxes(visible=False, row=3, col=2)
    
    fig.update_layout(
        title={
            'text': 'Data Distribution Overview',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 20}
        },
        width=1600,
        height=1400,
        showlegend=False
    )
    
    return fig

def generate_statistics_png(output_path=None,
                           width=1600, height=1400, dpi=100):
    """
    Generate statistics visualization directly as PNG using matplotlib
    
    Args:
        output_path: Path to save the PNG file
        width: Image width in pixels (default: 1600)
        height: Image height in pixels (default: 1400)
        dpi: Image resolution (default: 100)
    
    Returns:
        Path to the saved file
    """
    print("=" * 70)
    print("  Generating PNG (Using Matplotlib)")
    print("=" * 70)
    
    # Load data
    print(f"\n[1/3] Loading data...")
    tree, stats, df = load_taxonomy_and_stats()
    print(f"      ✓ {stats['num_classes']} classes, {stats['total_samples']} samples")
    
    # Create matplotlib figure
    print(f"\n[2/3] Building visualization...")
    fig, axes = plt.subplots(3, 2, figsize=(width/dpi, height/dpi), dpi=dpi)
    fig.suptitle('Data Distribution Overview', fontsize=20, fontweight='bold')
    
    # Class distribution
    class_counts = pd.Series(stats['class_distribution'])
    axes[0, 0].bar(class_counts.index, class_counts.values, color='#ff7f0e')
    axes[0, 0].set_title('Class Distribution', fontweight='bold')
    axes[0, 0].set_xlabel('Class')
    axes[0, 0].set_ylabel('Sample Count')
    axes[0, 0].tick_params(axis='x', rotation=45)
    for i, v in enumerate(class_counts.values):
        axes[0, 0].text(i, v, str(v), ha='center', va='bottom')
    
    # Top 10 Orders
    order_counts = pd.Series(stats['order_distribution']).sort_values(ascending=False).head(10)
    axes[0, 1].barh(order_counts.index, order_counts.values, color='#2ca02c')
    axes[0, 1].set_title('Top 10 Orders', fontweight='bold')
    axes[0, 1].set_xlabel('Sample Count')
    for i, v in enumerate(order_counts.values):
        axes[0, 1].text(v, i, str(v), va='center')
    
    # Top 10 Families
    family_counts = pd.Series(stats['family_distribution']).sort_values(ascending=False).head(10)
    axes[1, 0].barh(family_counts.index, family_counts.values, color='#d62728')
    axes[1, 0].set_title('Top 10 Families', fontweight='bold')
    axes[1, 0].set_xlabel('Sample Count')
    for i, v in enumerate(family_counts.values):
        axes[1, 0].text(v, i, str(v), va='center')
    
    # Top 15 Genera
    genus_counts = pd.Series(stats['genus_distribution']).sort_values(ascending=False).head(15)
    axes[1, 1].barh(genus_counts.index, genus_counts.values, color='#9467bd')
    axes[1, 1].set_title('Top 15 Genera', fontweight='bold')
    axes[1, 1].set_xlabel('Sample Count')
    for i, v in enumerate(genus_counts.values):
        axes[1, 1].text(v, i, str(v), va='center')
    
    # Top 15 Species
    if 'species_distribution' in stats:
        species_counts = pd.Series(stats['species_distribution']).sort_values(ascending=False).head(15)
    else:
        species_counts = df['species'].value_counts().sort_values(ascending=False).head(15)
    axes[2, 0].barh(species_counts.index, species_counts.values, color='#8c564b')
    axes[2, 0].set_title('Top 15 Species', fontweight='bold')
    axes[2, 0].set_xlabel('Sample Count')
    for i, v in enumerate(species_counts.values):
        axes[2, 0].text(v, i, str(v), va='center')
    
    # Hide empty subplot
    axes[2, 1].axis('off')
    
    plt.tight_layout()
    print(f"      ✓ Dashboard ready")
    
    # Save PNG
    output_path = Path(output_path) if output_path else REPORT_DIR / "data_distribution_overview.png"
    output_path.parent.mkdir(exist_ok=True, parents=True)
    
    print(f"\n[3/3] Exporting to PNG: {output_path}")
    print(f"      Size: {width}x{height}px @ {dpi} DPI")
    
    fig.savefig(str(output_path), dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    
    file_size = output_path.stat().st_size / (1024 * 1024)  # MB
    print(f"      ✓ PNG saved successfully!")
    print(f"      File size: {file_size:.2f} MB")
    
    print("\n" + "=" * 70)
    print("  Complete!")
    print("=" * 70)
    
    return output_path

def generate_statistics_html(output_path=None):
    """
    Generate statistics visualization as interactive HTML
    
    Args:
        output_path: Path to save the HTML file
    
    Returns:
        Path to the saved file
    """
    print("Generating interactive HTML...")
    tree, stats, df = load_taxonomy_and_stats()
    fig = create_statistics_dashboard(stats, df)
    
    output_path = Path(output_path) if output_path else REPORT_DIR / "data_distribution_overview.html"
    output_path.parent.mkdir(exist_ok=True, parents=True)
    fig.write_html(str(output_path))
    
    print(f"✓ Saved: {output_path}")
    return output_path

# For notebook usage
def show_statistics():
    """Display statistics dashboard in notebook"""
    tree, stats, df = load_taxonomy_and_stats()
    fig = create_statistics_dashboard(stats, df)
    fig.show()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Generate taxonomy visualization",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        '--views', 
        type=str, 
        default='statistics',
        choices=['statistics'],
        help='Type of view to generate (default: statistics)'
    )
    parser.add_argument(
        '--format', 
        type=str, 
        default='png',
        choices=['png', 'html'],
        help='Output format: png or html (default: png)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output file path (default: current run figures directory)'
    )
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("  Creating Data Distribution Visualization")
    print("=" * 70)
    
    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        if args.format == 'png':
            output_path = REPORT_DIR / "data_distribution_overview.png"
        else:
            output_path = REPORT_DIR / "data_distribution_overview.html"
    
    # Generate visualization
    if args.format == 'png':
        result = generate_statistics_png(output_path=output_path)
        if result:
            print(f"\n[OK] Statistics dashboard saved to: {result}")
        else:
            print(f"\n[ERROR] Failed to generate PNG")
            print(f"        The script tried to generate PNG but encountered an error.")
            print(f"        Common solutions:")
            print(f"        1. Install kaleido: pip install kaleido")
            print(f"        2. Install Chrome browser (kaleido requires Chrome)")
            print(f"        3. Try: plotly_get_chrome")
            print(f"        4. Or use HTML format: --format html")
            sys.exit(1)
    else:
        result = generate_statistics_html(output_path=output_path)
        print(f"\n[OK] Statistics dashboard saved to: {result}")
    
    print("\n" + "=" * 70)
    print("  Data Distribution Visualization Complete!")
    print("=" * 70)
    print(f"\nOutput file: {result}")
    print(f"This visualization shows the data distribution across taxonomic levels:")
    print(f"  - Class, Order, Family, Genus, and Species distributions")
    
    if args.format == 'html':
        print(f"\nOpen in your browser to view the distribution!")
