#!/usr/bin/env python3
"""
t-SNE Feature Distribution Comparison Visualization
Generate before/after transfer feature distribution comparison plots
"""
import os
import sys
import argparse
import numpy as np
import pickle
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib
from sklearn.manifold import TSNE
from matplotlib.lines import Line2D

# Set font and plotting style (journal quality)
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

# Journal-quality color scheme (colorblind-friendly)
COLORS = {
    'source': '#2E86AB',      # Deep blue - Source domain
    'target': '#A23B72',      # Purple-red - Target domain
    'background': '#F5F5F5',  # Light gray background
    'grid': '#E0E0E0'         # Grid line color
}


def load_features(feature_file: Path):
    """Load feature file"""
    with open(feature_file, 'rb') as f:
        data = pickle.load(f)
    return data


def compute_domain_distance(source_2d, target_2d):
    """
    Compute inter-domain distance using Wasserstein distance (Earth Mover's Distance) in t-SNE space
    
    Args:
        source_2d: Source domain 2D coordinates (N, 2)
        target_2d: Target domain 2D coordinates (M, 2)
    
    Returns:
        Wasserstein distance value (float)
    """
    try:
        # Try using POT (Python Optimal Transport) library for Wasserstein distance
        import ot
        # Compute Wasserstein distance using POT
        # Create uniform distributions for both domains
        n_source = len(source_2d)
        n_target = len(target_2d)
        
        # Uniform weights
        a = np.ones(n_source) / n_source
        b = np.ones(n_target) / n_target
        
        # Compute cost matrix (Euclidean distance matrix)
        M = ot.dist(source_2d, target_2d, metric='euclidean')
        
        # Compute Wasserstein distance (2-Wasserstein)
        wasserstein_dist = ot.emd2(a, b, M)
        
        return wasserstein_dist
    except ImportError:
        # Fallback: use scipy's wasserstein_distance for 1D, then average
        from scipy.stats import wasserstein_distance
        # Compute Wasserstein distance for each dimension and average
        wd_x = wasserstein_distance(source_2d[:, 0], target_2d[:, 0])
        wd_y = wasserstein_distance(source_2d[:, 1], target_2d[:, 1])
        # Return Euclidean norm of the 2D Wasserstein distances
        wasserstein_dist = np.sqrt(wd_x**2 + wd_y**2)
        return wasserstein_dist


def plot_tsne_comparison(source_features, target_features, source_labels, target_labels,
                        source_size, target_size, title, ax=None):
    """
    Plot t-SNE feature distribution comparison
    
    Args:
        source_features: Source domain features
        target_features: Target domain features
        source_labels: Source domain labels
        target_labels: Target domain labels
        source_size: Source domain size (e.g., '0.5')
        target_size: Target domain size (e.g., '0.35')
        title: Subplot title
        ax: Matplotlib axes (if None, creates new figure)
    
    Returns:
        Inter-domain distance (float)
    """
    # Merge features and labels
    all_features = np.concatenate([source_features, target_features], axis=0)
    all_labels = np.concatenate([source_labels, target_labels], axis=0)
    all_domains = np.concatenate([
        np.zeros(len(source_features), dtype=int),
        np.ones(len(target_features), dtype=int)
    ], axis=0)
    
    print(f"\nPreparing to plot: {title}")
    print(f"Feature shape: {all_features.shape}")
    
    # PCA dimensionality reduction if features are too high-dimensional
    if all_features.shape[1] > 50:
        from sklearn.decomposition import PCA
        print("Features are high-dimensional, performing PCA first...")
        pca = PCA(n_components=50, random_state=42)
        all_features_pca = pca.fit_transform(all_features)
        print(f"After PCA: {all_features_pca.shape}, explained variance ratio: {pca.explained_variance_ratio_.sum():.4f}")
    else:
        all_features_pca = all_features
    
    # t-SNE dimensionality reduction
    print("Performing t-SNE dimensionality reduction...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000, verbose=0)
    features_2d = tsne.fit_transform(all_features_pca)
    
    # Create or use provided axes
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))
        fig.patch.set_facecolor('white')
    
    ax.set_facecolor(COLORS['background'])
    
    # Separate source and target domain 2D features
    source_2d = features_2d[all_domains == 0]
    target_2d = features_2d[all_domains == 1]
    source_labels_plot = all_labels[all_domains == 0]
    target_labels_plot = all_labels[all_domains == 1]
    
    # Compute inter-domain Wasserstein distance
    domain_distance = compute_domain_distance(source_2d, target_2d)
    print(f"Inter-domain Wasserstein distance: {domain_distance:.4f}")
    
    # Plot source domain features (circular markers, no border)
    for label_id in np.unique(source_labels_plot):
        mask = source_labels_plot == label_id
        ax.scatter(source_2d[mask, 0], source_2d[mask, 1], 
                  c=COLORS['source'], marker='o', s=40, alpha=0.7, 
                  edgecolors='none', label='')
    
    # Plot target domain features (circular markers, no border)
    for label_id in np.unique(target_labels_plot):
        mask = target_labels_plot == label_id
        ax.scatter(target_2d[mask, 0], target_2d[mask, 1], 
                  c=COLORS['target'], marker='o', s=40, alpha=0.7, 
                  edgecolors='none', label='')
    
    # Set plot properties
    ax.set_xlabel('t-SNE Dimension 1', fontsize=12, fontweight='bold')
    ax.set_ylabel('t-SNE Dimension 2', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=13, fontweight='bold', pad=15)
    
    # Add inter-domain Wasserstein distance text in upper left corner
    ax.text(0.02, 0.98, f'Wasserstein Distance: {domain_distance:.3f}', 
           transform=ax.transAxes, fontsize=11, fontweight='bold',
           verticalalignment='top', horizontalalignment='left',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray'))
    
    # Add legend (only domain labels)
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=COLORS['source'], 
               markersize=8, markeredgecolor='none', label='Source Domain'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=COLORS['target'], 
               markersize=8, markeredgecolor='none', label='Target Domain')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10, 
             framealpha=0.9, fancybox=True, shadow=True)
    
    # Set grid
    ax.grid(True, linestyle='--', alpha=0.3, color=COLORS['grid'])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)
    
    return domain_distance


def plot_combined_comparison(direction: str, base_mlp_file: Path, domain_adapt_file: Path,
                            output_file: Path, source_size: str, target_size: str):
    """
    Plot combined comparison (before and after transfer as subplots)
    
    Args:
        direction: Transfer direction (e.g., '05_to_035')
        base_mlp_file: Path to base MLP features file (before transfer)
        domain_adapt_file: Path to domain adaptation features file (after transfer)
        output_file: Output file path
        source_size: Source domain size
        target_size: Target domain size
    """
    # Load features
    print(f"\n{'='*60}")
    print(f"Generating combined plot: {direction}")
    print(f"{'='*60}")
    
    base_data = load_features(base_mlp_file)
    adapt_data = load_features(domain_adapt_file)
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    fig.patch.set_facecolor('white')
    
    # Plot pre-transfer (base MLP)
    print("Plotting pre-transfer...")
    distance_before = plot_tsne_comparison(
        base_data['source_features'], base_data['target_features'],
        base_data['source_labels'], base_data['target_labels'],
        source_size, target_size, 
        'Pre-transfer (Fused Feature Distribution)',
        ax=ax1
    )
    
    # Plot post-transfer (domain adaptation)
    print("Plotting post-transfer...")
    distance_after = plot_tsne_comparison(
        adapt_data['source_features'], adapt_data['target_features'],
        adapt_data['source_labels'], adapt_data['target_labels'],
        source_size, target_size,
        'Post-transfer (Fused Feature Distribution)',
        ax=ax2
    )
    
    # Add main title (using parentheses for transfer direction)
    fig.suptitle(f't-SNE Feature Distribution Comparison ({source_size} → {target_size})', 
                fontsize=16, fontweight='bold', y=1.00)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"Combined plot saved to: {output_file}")
    print(f"  Pre-transfer Wasserstein distance: {distance_before:.4f}")
    print(f"  Post-transfer Wasserstein distance: {distance_after:.4f}")
    print(f"  Distance reduction: {distance_before - distance_after:.4f} ({((distance_before - distance_after) / distance_before * 100):.2f}%)")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='t-SNE Feature Distribution Comparison Visualization')
    parser.add_argument('--features_dir', type=str, default='features',
                       help='Feature files directory')
    parser.add_argument('--output_dir', type=str, default='tsne_visualizations',
                       help='Output directory')
    
    args = parser.parse_args()
    
    base_dir = Path(__file__).parent
    features_dir = Path(args.features_dir)
    if not features_dir.is_absolute():
        features_dir = base_dir / features_dir
    
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = base_dir / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("t-SNE Feature Distribution Comparison Visualization")
    print("=" * 60)
    print(f"Features directory: {features_dir}")
    print(f"Output directory: {output_dir}")
    print("=" * 60)
    
    try:
        # Direction 1: 0.5 → 0.35
        print("\n" + "="*60)
        print("Generating combined plot 1: 0.5 → 0.35")
        print("="*60)
        base_mlp_file_1 = features_dir / 'base_mlp_features_05_to_035.pkl'
        v3_file = features_dir / 'v3_features_05_to_035.pkl'
        
        if base_mlp_file_1.exists() and v3_file.exists():
            plot_combined_comparison(
                '05_to_035',
                base_mlp_file_1,
                v3_file,
                output_dir / 't-SNE_Comparison_05_to_035.png',
                '0.5', '0.35'
            )
        else:
            print(f"Warning: Feature files missing for 0.5 → 0.35")
            if not base_mlp_file_1.exists():
                print(f"  Missing: {base_mlp_file_1}")
            if not v3_file.exists():
                print(f"  Missing: {v3_file}")
        
        # Direction 2: 0.35 → 0.5
        print("\n" + "="*60)
        print("Generating combined plot 2: 0.35 → 0.5")
        print("="*60)
        base_mlp_file_2 = features_dir / 'base_mlp_features_035_to_05.pkl'
        v7_file = features_dir / 'v7_features_035_to_05.pkl'
        
        if base_mlp_file_2.exists() and v7_file.exists():
            plot_combined_comparison(
                '035_to_05',
                base_mlp_file_2,
                v7_file,
                output_dir / 't-SNE_Comparison_035_to_05.png',
                '0.35', '0.5'
            )
        else:
            print(f"Warning: Feature files missing for 0.35 → 0.5")
            if not base_mlp_file_2.exists():
                print(f"  Missing: {base_mlp_file_2}")
            if not v7_file.exists():
                print(f"  Missing: {v7_file}")
        
        print(f"\n{'='*60}")
        print("t-SNE Visualization Complete!")
        print(f"Plots saved in: {output_dir}")
        print(f"{'='*60}")
        
    except Exception as e:
        print(f"\nVisualization failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
