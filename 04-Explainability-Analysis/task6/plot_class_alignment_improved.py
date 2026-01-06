#!/usr/bin/env python3
"""
Improved Class-wise Alignment Visualization
Only analyze classes that exist in both domains
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
from sklearn.decomposition import PCA
from scipy.spatial.distance import pdist, cdist
import seaborn as sns
from matplotlib.lines import Line2D

# Set font and plotting style (journal quality)
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

# Journal-quality color scheme
COLORS = {
    'pre': '#2E86AB',       # Deep blue - Pre-transfer
    'post': '#A23B72',      # Purple-red - Post-transfer
    'missing': '#D3D3D3',   # Light gray - Missing class
    'background': '#F5F5F5', # Light gray background
    'grid': '#E0E0E0'        # Grid line color
}

CLASS_COLORS = ['#66C2A5', '#FC8D62', '#8DA0CB', '#E78AC3', '#A6D854']
CLASS_COLORS_DARK = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']


def load_features(feature_file: Path):
    """Load feature file"""
    with open(feature_file, 'rb') as f:
        data = pickle.load(f)
    return data


def get_common_classes(source_labels, target_labels):
    """
    Get classes that exist in both domains
    
    Returns:
        common_classes: numpy array of common class IDs
        source_only: set of classes only in source
        target_only: set of classes only in target
    """
    source_classes = set(np.unique(source_labels))
    target_classes = set(np.unique(target_labels))
    
    common_classes = np.array(sorted(list(source_classes & target_classes)))
    source_only = source_classes - target_classes
    target_only = target_classes - source_classes
    
    return common_classes, source_only, target_only


def compute_t_sne_2d(features, random_state=42):
    """Compute 2D t-SNE coordinates"""
    if features.shape[1] > 50:
        pca = PCA(n_components=50, random_state=random_state)
        features_pca = pca.fit_transform(features)
    else:
        features_pca = features
    
    tsne = TSNE(n_components=2, random_state=random_state, perplexity=30, max_iter=1000, verbose=0)
    features_2d = tsne.fit_transform(features_pca)
    return features_2d


def compute_class_metrics_common(features_2d, labels, common_classes):
    """
    Compute intra-class and inter-class distances only for common classes
    
    Args:
        features_2d: 2D feature coordinates (N, 2)
        labels: Class labels (N,)
        common_classes: Array of common class IDs
    
    Returns:
        Dictionary with metrics for each common class
    """
    metrics = {}
    
    for cls in common_classes:
        # Get samples of this class
        class_mask = labels == cls
        class_samples = features_2d[class_mask]
        
        # Get samples of all other common classes
        other_mask = np.isin(labels, common_classes) & (labels != cls)
        other_samples = features_2d[other_mask]
        
        # Compute intra-class distance
        if len(class_samples) > 1:
            intra_dist = np.mean(pdist(class_samples, metric='euclidean'))
        else:
            intra_dist = 0.0
        
        # Compute inter-class distance (to other common classes only)
        if len(other_samples) > 0:
            inter_dist = np.mean(cdist(class_samples, other_samples, metric='euclidean'))
        else:
            inter_dist = 0.0
        
        metrics[cls] = {
            'intra_class_distance': intra_dist,
            'inter_class_distance': inter_dist,
            'n_samples': len(class_samples)
        }
    
    return metrics


def plot_improved_scatter(base_data, adapt_data, source_size, target_size, output_file):
    """
    Improved scatter plot - only showing common classes with clear annotation of missing classes
    """
    print(f"\nGenerating improved scatter plot: {source_size} → {target_size}")
    
    # Get common classes
    pre_source_classes = set(np.unique(base_data['source_labels']))
    pre_target_classes = set(np.unique(base_data['target_labels']))
    post_source_classes = set(np.unique(adapt_data['source_labels']))
    post_target_classes = set(np.unique(adapt_data['target_labels']))
    
    pre_common = sorted(list(pre_source_classes & pre_target_classes))
    post_common = sorted(list(post_source_classes & post_target_classes))
    
    print(f"Pre-transfer common classes: {pre_common}")
    print(f"Post-transfer common classes: {post_common}")
    
    # Filter data to only include common classes
    pre_source_mask = np.isin(base_data['source_labels'], pre_common)
    pre_target_mask = np.isin(base_data['target_labels'], pre_common)
    
    pre_source_features_common = base_data['source_features'][pre_source_mask]
    pre_target_features_common = base_data['target_features'][pre_target_mask]
    pre_source_labels_common = base_data['source_labels'][pre_source_mask]
    pre_target_labels_common = base_data['target_labels'][pre_target_mask]
    
    post_source_mask = np.isin(adapt_data['source_labels'], post_common)
    post_target_mask = np.isin(adapt_data['target_labels'], post_common)
    
    post_source_features_common = adapt_data['source_features'][post_source_mask]
    post_target_features_common = adapt_data['target_features'][post_target_mask]
    post_source_labels_common = adapt_data['source_labels'][post_source_mask]
    post_target_labels_common = adapt_data['target_labels'][post_target_mask]
    
    # Combine features for t-SNE (using common classes only)
    pre_features_combined = np.concatenate([pre_source_features_common, pre_target_features_common], axis=0)
    pre_labels_combined = np.concatenate([pre_source_labels_common, pre_target_labels_common], axis=0)
    
    post_features_combined = np.concatenate([post_source_features_common, post_target_features_common], axis=0)
    post_labels_combined = np.concatenate([post_source_labels_common, post_target_labels_common], axis=0)
    
    # Compute t-SNE
    pre_2d = compute_t_sne_2d(pre_features_combined, random_state=42)
    post_2d = compute_t_sne_2d(post_features_combined, random_state=42)
    
    # Compute metrics for common classes
    pre_metrics = compute_class_metrics_common(pre_2d, pre_labels_combined, pre_common)
    post_metrics = compute_class_metrics_common(post_2d, post_labels_combined, post_common)
    
    # Separate source and target in 2D space
    n_pre_source = len(pre_source_features_common)
    n_post_source = len(post_source_features_common)
    
    pre_source_2d = pre_2d[:n_pre_source]
    pre_target_2d = pre_2d[n_pre_source:]
    pre_source_labels_2d = pre_labels_combined[:n_pre_source]
    pre_target_labels_2d = pre_labels_combined[n_pre_source:]
    
    post_source_2d = post_2d[:n_post_source]
    post_target_2d = post_2d[n_post_source:]
    post_source_labels_2d = post_labels_combined[:n_post_source]
    post_target_labels_2d = post_labels_combined[n_post_source:]
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    fig.patch.set_facecolor('white')
    
    # Plot 1: Pre-transfer
    ax1.set_facecolor(COLORS['background'])
    
    for cls in pre_common:
        color_idx = int(cls) % len(CLASS_COLORS)
        color = CLASS_COLORS[color_idx]
        
        # Source domain samples
        source_mask = pre_source_labels_2d == cls
        if source_mask.any():
            ax1.scatter(pre_source_2d[source_mask, 0], pre_source_2d[source_mask, 1],
                       c=[color], marker='o', s=40, alpha=0.6, edgecolors='none',
                       label=f'Source Class {int(cls)}' if cls == pre_common[0] else '')
        
        # Target domain samples
        target_mask = pre_target_labels_2d == cls
        if target_mask.any():
            ax1.scatter(pre_target_2d[target_mask, 0], pre_target_2d[target_mask, 1],
                       c=[color], marker='s', s=40, alpha=0.6, edgecolors='none',
                       label=f'Target Class {int(cls)}' if cls == pre_common[0] else '')
    
    ax1.set_xlabel('t-SNE Dimension 1', fontsize=12, fontweight='bold')
    ax1.set_ylabel('t-SNE Dimension 2', fontsize=12, fontweight='bold')
    ax1.set_title('Pre-transfer\n(Common Classes Only)', fontsize=13, fontweight='bold', pad=15)
    ax1.legend(loc='best', fontsize=9, framealpha=0.9)
    ax1.grid(True, linestyle='--', alpha=0.3, color=COLORS['grid'])
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    # Add note about missing classes
    missing_pre = sorted(list((pre_source_classes | pre_target_classes) - set(pre_common)))
    if missing_pre:
        ax1.text(0.02, 0.02, f'Note: Class(es) {missing_pre} not present in both domains',
                transform=ax1.transAxes, fontsize=9, style='italic',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.3))
    
    # Plot 2: Post-transfer
    ax2.set_facecolor(COLORS['background'])
    
    for cls in post_common:
        color_idx = int(cls) % len(CLASS_COLORS)
        color = CLASS_COLORS[color_idx]
        
        # Source domain samples
        source_mask = post_source_labels_2d == cls
        if source_mask.any():
            ax2.scatter(post_source_2d[source_mask, 0], post_source_2d[source_mask, 1],
                       c=[color], marker='o', s=40, alpha=0.6, edgecolors='none')
        
        # Target domain samples
        target_mask = post_target_labels_2d == cls
        if target_mask.any():
            ax2.scatter(post_target_2d[target_mask, 0], post_target_2d[target_mask, 1],
                       c=[color], marker='s', s=40, alpha=0.6, edgecolors='none')
    
    ax2.set_xlabel('t-SNE Dimension 1', fontsize=12, fontweight='bold')
    ax2.set_ylabel('t-SNE Dimension 2', fontsize=12, fontweight='bold')
    ax2.set_title('Post-transfer\n(Common Classes Only)', fontsize=13, fontweight='bold', pad=15)
    ax2.grid(True, linestyle='--', alpha=0.3, color=COLORS['grid'])
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    
    # Add note about missing classes
    missing_post = sorted(list((post_source_classes | post_target_classes) - set(post_common)))
    if missing_post:
        ax2.text(0.02, 0.02, f'Note: Class(es) {missing_post} not present in both domains',
                transform=ax2.transAxes, fontsize=9, style='italic',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.3))
    
    fig.suptitle(f'Class-wise Feature Alignment (Common Classes Only) ({source_size} → {target_size})',
                fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    # Print summary
    print(f"\nClass-wise Metrics (Common Classes Only):")
    print(f"{'Class':<10} {'Pre-transfer':<30} {'Post-transfer':<30} {'Change':<25}")
    print(f"{'':<10} {'Intra':<12} {'Inter':<12} {'Intra':<12} {'Inter':<12} {'Intra Δ%':<12} {'Inter Δ%':<12}")
    print("-" * 100)
    
    for cls in pre_common:
        if cls in pre_metrics and cls in post_metrics:
            pre_intra = pre_metrics[cls]['intra_class_distance']
            pre_inter = pre_metrics[cls]['inter_class_distance']
            post_intra = post_metrics[cls]['intra_class_distance']
            post_inter = post_metrics[cls]['inter_class_distance']
            
            intra_change = ((post_intra - pre_intra) / pre_intra * 100) if pre_intra > 0 else 0
            inter_change = ((post_inter - pre_inter) / pre_inter * 100) if pre_inter > 0 else 0
            
            print(f"Class {int(cls):<7} {pre_intra:>10.3f} {pre_inter:>10.3f}   "
                  f"{post_intra:>10.3f} {post_inter:>10.3f}   "
                  f"{intra_change:>10.1f}% {inter_change:>10.1f}%")
    
    print(f"\nPlot saved to: {output_file}")


def plot_common_class_comparison(base_data, adapt_data, source_size, target_size, output_file):
    """
    Comparison plot focusing on common classes - intra vs inter distance scatter
    """
    print(f"\nGenerating common class comparison: {source_size} → {target_size}")
    
    # Get common classes
    pre_source_classes = set(np.unique(base_data['source_labels']))
    pre_target_classes = set(np.unique(base_data['target_labels']))
    post_source_classes = set(np.unique(adapt_data['source_labels']))
    post_target_classes = set(np.unique(adapt_data['target_labels']))
    
    pre_common = sorted(list(pre_source_classes & pre_target_classes))
    post_common = sorted(list(post_source_classes & post_target_classes))
    all_common = sorted(list(set(pre_common) & set(post_common)))
    
    print(f"Common classes for analysis: {all_common}")
    
    # Filter to common classes
    pre_source_mask = np.isin(base_data['source_labels'], all_common)
    pre_target_mask = np.isin(base_data['target_labels'], all_common)
    post_source_mask = np.isin(adapt_data['source_labels'], all_common)
    post_target_mask = np.isin(adapt_data['target_labels'], all_common)
    
    # Combine features
    pre_features = np.concatenate([
        base_data['source_features'][pre_source_mask],
        base_data['target_features'][pre_target_mask]
    ], axis=0)
    pre_labels = np.concatenate([
        base_data['source_labels'][pre_source_mask],
        base_data['target_labels'][pre_target_mask]
    ], axis=0)
    
    post_features = np.concatenate([
        adapt_data['source_features'][post_source_mask],
        adapt_data['target_features'][post_target_mask]
    ], axis=0)
    post_labels = np.concatenate([
        adapt_data['source_labels'][post_source_mask],
        adapt_data['target_labels'][post_target_mask]
    ], axis=0)
    
    # Compute t-SNE
    pre_2d = compute_t_sne_2d(pre_features, random_state=42)
    post_2d = compute_t_sne_2d(post_features, random_state=42)
    
    # Compute metrics
    pre_metrics = compute_class_metrics_common(pre_2d, pre_labels, all_common)
    post_metrics = compute_class_metrics_common(post_2d, post_labels, all_common)
    
    # Prepare data for scatter plot
    pre_intra = [pre_metrics[cls]['intra_class_distance'] for cls in all_common]
    pre_inter = [pre_metrics[cls]['inter_class_distance'] for cls in all_common]
    post_intra = [post_metrics[cls]['intra_class_distance'] for cls in all_common]
    post_inter = [post_metrics[cls]['inter_class_distance'] for cls in all_common]
    
    class_labels = [f'Class {int(c)}' for c in all_common]
    
    fig, ax = plt.subplots(figsize=(10, 8))
    fig.patch.set_facecolor('white')
    ax.set_facecolor(COLORS['background'])
    
    # Plot pre-transfer points
    scatter_pre = ax.scatter(pre_intra, pre_inter, c=COLORS['pre'], s=250, 
                            alpha=0.8, marker='o', edgecolors='white', linewidths=2.5,
                            label='Pre-transfer', zorder=3)
    
    # Plot post-transfer points
    scatter_post = ax.scatter(post_intra, post_inter, c=COLORS['post'], s=250,
                             alpha=0.8, marker='s', edgecolors='white', linewidths=2.5,
                             label='Post-transfer', zorder=3)
    
    # Draw arrows
    for i, cls in enumerate(all_common):
        dx = post_intra[i] - pre_intra[i]
        dy = post_inter[i] - pre_inter[i]
        if abs(dx) > 0.1 or abs(dy) > 0.1:
            ax.annotate('', xy=(post_intra[i], post_inter[i]), 
                       xytext=(pre_intra[i], pre_inter[i]),
                       arrowprops=dict(arrowstyle='->', color='#FF6B35', lw=2.5, 
                                     alpha=0.7, zorder=2, connectionstyle='arc3,rad=0.1'))
        
        # Add class labels
        ax.text(pre_intra[i], pre_inter[i], f'  {int(cls)}', 
               fontsize=10, va='bottom', ha='left', weight='bold',
               bbox=dict(boxstyle='round,pad=0.4', facecolor='white', 
                        alpha=0.85, edgecolor=COLORS['pre'], linewidth=1.5))
    
    ax.set_xlabel('Intra-class Distance (smaller is better)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Inter-class Distance (larger is better)', fontsize=13, fontweight='bold')
    ax.set_title(f'Class-wise Distance Comparison (Common Classes Only)\n({source_size} → {target_size})',
                fontsize=14, fontweight='bold', pad=15)
    ax.legend(loc='best', fontsize=11, framealpha=0.9, fancybox=True, shadow=True, markerscale=1.2)
    ax.grid(True, linestyle='--', alpha=0.4, color=COLORS['grid'])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)
    
    # Add note
    missing_info = []
    if pre_source_classes - pre_target_classes:
        missing_info.append(f"Pre: Class {sorted(list(pre_source_classes - pre_target_classes))} missing in target")
    if post_source_classes - post_target_classes:
        missing_info.append(f"Post: Class {sorted(list(post_source_classes - post_target_classes))} missing in target")
    
    if missing_info:
        note_text = '\n'.join(missing_info)
        ax.text(0.98, 0.02, note_text, transform=ax.transAxes, fontsize=8, 
               ha='right', va='bottom', style='italic',
               bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.3))
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"Plot saved to: {output_file}")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Improved Class Alignment Visualization')
    parser.add_argument('--features_dir', type=str, default='features',
                       help='Feature files directory')
    parser.add_argument('--output_dir', type=str, default='class_alignment_improved',
                       help='Output directory')
    parser.add_argument('--direction', type=str, choices=['05_to_035', '035_to_05', 'both'],
                       default='both', help='Which transfer direction(s) to visualize')
    
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
    print("Improved Class Alignment Visualization")
    print("(Only analyzing common classes)")
    print("=" * 60)
    
    directions_to_process = []
    if args.direction in ['05_to_035', 'both']:
        directions_to_process.append(('05_to_035', '0.5', '0.35'))
    if args.direction in ['035_to_05', 'both']:
        directions_to_process.append(('035_to_05', '0.35', '0.5'))
    
    try:
        for direction_key, source_size, target_size in directions_to_process:
            size_map = {'0.5': '05', '0.35': '035'}
            source_key = size_map[source_size]
            target_key = size_map[target_size]
            
            base_file = features_dir / f'base_mlp_features_{source_key}_to_{target_key}.pkl'
            adapt_file = features_dir / f'v{3 if direction_key == "05_to_035" else 7}_features_{source_key}_to_{target_key}.pkl'
            
            if not base_file.exists() or not adapt_file.exists():
                print(f"Warning: Feature files missing for {source_size} → {target_size}")
                continue
            
            base_data = load_features(base_file)
            adapt_data = load_features(adapt_file)
            
            # Plot 1: Improved scatter
            plot_improved_scatter(
                base_data, adapt_data, source_size, target_size,
                output_dir / f'Improved_Scatter_{source_size}_to_{target_size}.png'
            )
            
            # Plot 2: Common class comparison
            plot_common_class_comparison(
                base_data, adapt_data, source_size, target_size,
                output_dir / f'Common_Class_Comparison_{source_size}_to_{target_size}.png'
            )
        
        print(f"\n{'='*60}")
        print("Improved visualization complete!")
        print(f"Results saved in: {output_dir}")
        print(f"{'='*60}")
        
    except Exception as e:
        print(f"\nVisualization failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()




