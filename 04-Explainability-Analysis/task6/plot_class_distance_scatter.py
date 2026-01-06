#!/usr/bin/env python3
"""
Class-wise Distance Scatter Plot
Visualize intra-class and inter-class distances for each class before and after transfer
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
from scipy.spatial.distance import pdist, cdist
from matplotlib.patches import FancyBboxPatch

# Set font and plotting style (journal quality)
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

# Journal-quality color scheme
COLORS = {
    'pre': '#2E86AB',       # Deep blue - Pre-transfer
    'post': '#A23B72',      # Purple-red - Post-transfer
    'arrow': '#FF6B35',     # Orange - Arrow color
    'background': '#F5F5F5', # Light gray background
    'grid': '#E0E0E0'        # Grid line color
}

# Class labels (adjust based on your actual labels)
CLASS_LABELS = ['Class 0', 'Class 1', 'Class 2', 'Class 3', 'Class 4']


def load_features(feature_file: Path):
    """Load feature file"""
    with open(feature_file, 'rb') as f:
        data = pickle.load(f)
    return data


def compute_t_sne_2d(features):
    """
    Compute 2D t-SNE coordinates for features
    
    Args:
        features: Feature array (N, feature_dim)
    
    Returns:
        2D coordinates (N, 2)
    """
    # PCA dimensionality reduction if features are too high-dimensional
    if features.shape[1] > 50:
        from sklearn.decomposition import PCA
        pca = PCA(n_components=50, random_state=42)
        features_pca = pca.fit_transform(features)
    else:
        features_pca = features
    
    # t-SNE dimensionality reduction
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000, verbose=0)
    features_2d = tsne.fit_transform(features_pca)
    
    return features_2d


def compute_intra_class_distance(class_samples_2d):
    """
    Compute intra-class distance (average distance within class)
    
    Args:
        class_samples_2d: 2D coordinates of samples in one class (N, 2)
    
    Returns:
        Average intra-class distance (float)
    """
    if len(class_samples_2d) < 2:
        return 0.0
    
    # Compute pairwise distances within the class
    pairwise_distances = pdist(class_samples_2d, metric='euclidean')
    
    # Return mean distance
    return np.mean(pairwise_distances)


def compute_inter_class_distance(class_samples_2d, other_classes_samples_2d):
    """
    Compute inter-class distance (average distance to other classes)
    
    Args:
        class_samples_2d: 2D coordinates of samples in current class (N, 2)
        other_classes_samples_2d: 2D coordinates of samples in all other classes (M, 2)
    
    Returns:
        Average inter-class distance (float)
    """
    if len(other_classes_samples_2d) == 0:
        return 0.0
    
    # Compute distances from current class to all other classes
    cross_distances = cdist(class_samples_2d, other_classes_samples_2d, metric='euclidean')
    
    # Return mean distance
    return np.mean(cross_distances)


def compute_class_distances(source_features, target_features, source_labels, target_labels):
    """
    Compute intra-class and inter-class distances for all classes
    
    Args:
        source_features: Source domain features
        target_features: Target domain features
        source_labels: Source domain labels
        target_labels: Target domain labels
    
    Returns:
        Dictionary with class-wise statistics
    """
    # Combine all features and labels
    all_features = np.concatenate([source_features, target_features], axis=0)
    all_labels = np.concatenate([source_labels, target_labels], axis=0)
    
    # Compute 2D t-SNE coordinates
    features_2d = compute_t_sne_2d(all_features)
    
    # Get unique classes
    unique_classes = np.unique(all_labels)
    n_classes = len(unique_classes)
    
    # Compute distances for each class
    class_stats = {}
    
    for class_id in unique_classes:
        # Get samples of this class
        class_mask = all_labels == class_id
        class_samples_2d = features_2d[class_mask]
        
        # Get samples of all other classes
        other_mask = all_labels != class_id
        other_samples_2d = features_2d[other_mask]
        
        # Compute intra-class distance
        intra_dist = compute_intra_class_distance(class_samples_2d)
        
        # Compute inter-class distance
        inter_dist = compute_inter_class_distance(class_samples_2d, other_samples_2d)
        
        class_stats[class_id] = {
            'intra_class_distance': intra_dist,
            'inter_class_distance': inter_dist,
            'n_samples': len(class_samples_2d)
        }
    
    return class_stats, unique_classes


def plot_class_distance_scatter(base_data, adapt_data, source_size, target_size, 
                                output_file, class_labels=None):
    """
    Plot class-wise distance scatter plot (before and after transfer)
    
    Args:
        base_data: Pre-transfer feature data dictionary
        adapt_data: Post-transfer feature data dictionary
        source_size: Source domain size
        target_size: Target domain size
        output_file: Output file path
        class_labels: Optional list of class label names
    """
    print(f"\n{'='*60}")
    print(f"Computing class distances: {source_size} → {target_size}")
    print(f"{'='*60}")
    
    # Compute distances for pre-transfer
    print("Computing pre-transfer class distances...")
    pre_stats, unique_classes = compute_class_distances(
        base_data['source_features'],
        base_data['target_features'],
        base_data['source_labels'],
        base_data['target_labels']
    )
    
    # Compute distances for post-transfer
    print("Computing post-transfer class distances...")
    post_stats, _ = compute_class_distances(
        adapt_data['source_features'],
        adapt_data['target_features'],
        adapt_data['source_labels'],
        adapt_data['target_labels']
    )
    
    # Prepare data for plotting
    pre_intra = [pre_stats[cls]['intra_class_distance'] for cls in unique_classes]
    pre_inter = [pre_stats[cls]['inter_class_distance'] for cls in unique_classes]
    post_intra = [post_stats[cls]['intra_class_distance'] for cls in unique_classes]
    post_inter = [post_stats[cls]['inter_class_distance'] for cls in unique_classes]
    
    # Use provided class labels or generate default ones
    if class_labels is None or len(class_labels) != len(unique_classes):
        plot_labels = [f'Class {int(cls)}' for cls in unique_classes]
    else:
        plot_labels = [class_labels[int(cls)] for cls in unique_classes]
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    fig.patch.set_facecolor('white')
    
    # Plot 1: Scatter plot with arrows showing movement
    ax1.set_facecolor(COLORS['background'])
    
    # Plot pre-transfer points
    scatter_pre = ax1.scatter(pre_intra, pre_inter, c=COLORS['pre'], s=250, 
                             alpha=0.8, marker='o', edgecolors='white', linewidths=2.5,
                             label='Pre-transfer', zorder=3)
    
    # Plot post-transfer points
    scatter_post = ax1.scatter(post_intra, post_inter, c=COLORS['post'], s=250,
                              alpha=0.8, marker='s', edgecolors='white', linewidths=2.5,
                              label='Post-transfer', zorder=3)
    
    # Draw arrows from pre to post
    for i, cls in enumerate(unique_classes):
        dx = post_intra[i] - pre_intra[i]
        dy = post_inter[i] - pre_inter[i]
        # Only draw arrow if movement is significant
        if abs(dx) > 0.1 or abs(dy) > 0.1:
            ax1.annotate('', xy=(post_intra[i], post_inter[i]), 
                        xytext=(pre_intra[i], pre_inter[i]),
                        arrowprops=dict(arrowstyle='->', color=COLORS['arrow'], 
                                      lw=2.5, alpha=0.7, zorder=2,
                                      connectionstyle='arc3,rad=0.1'))
    
    # Add class labels near points
    for i, (label, cls) in enumerate(zip(plot_labels, unique_classes)):
        # Label near pre-transfer point
        ax1.text(pre_intra[i], pre_inter[i], f'  {label}', 
                fontsize=10, va='bottom', ha='left', 
                bbox=dict(boxstyle='round,pad=0.4', facecolor='white', 
                         alpha=0.85, edgecolor=COLORS['pre'], linewidth=1.5),
                weight='bold')
    
    ax1.set_xlabel('Intra-class Distance (smaller is better)', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Inter-class Distance (larger is better)', fontsize=13, fontweight='bold')
    ax1.set_title('Class-wise Distance Comparison with Movement', fontsize=14, fontweight='bold', pad=15)
    ax1.legend(loc='best', fontsize=11, framealpha=0.9, fancybox=True, shadow=True, markerscale=1.2)
    ax1.grid(True, linestyle='--', alpha=0.4, color=COLORS['grid'])
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['left'].set_linewidth(1.5)
    ax1.spines['bottom'].set_linewidth(1.5)
    
    # Add diagonal reference lines (ideal direction)
    x_min, x_max = ax1.get_xlim()
    y_min, y_max = ax1.get_ylim()
    # Ideal movement: left and up (toward bottom-left to top-right)
    ax1.text(0.98, 0.02, 'Ideal: ←↓ (lower intra) and ↑→ (higher inter)', 
            transform=ax1.transAxes, fontsize=9, ha='right', va='bottom',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.3),
            style='italic')
    
    # Plot 2: Detailed comparison bar chart
    ax2.set_facecolor(COLORS['background'])
    
    x_pos = np.arange(len(unique_classes))
    width = 0.35
    
    # Plot bars for intra-class distance (left axis)
    bars1 = ax2.bar(x_pos - width/2, pre_intra, width, label='Pre-transfer (Intra)', 
                   color=COLORS['pre'], alpha=0.7, edgecolor='white', linewidth=1.5)
    bars2 = ax2.bar(x_pos + width/2, post_intra, width, label='Post-transfer (Intra)', 
                   color=COLORS['post'], alpha=0.7, edgecolor='white', linewidth=1.5)
    
    # Create second y-axis for inter-class distance
    ax2_twin = ax2.twinx()
    bars3 = ax2_twin.bar(x_pos - width/2, pre_inter, width, label='Pre-transfer (Inter)', 
                        color=COLORS['pre'], alpha=0.4, hatch='///', edgecolor='white', linewidth=1.5)
    bars4 = ax2_twin.bar(x_pos + width/2, post_inter, width, label='Post-transfer (Inter)', 
                        color=COLORS['post'], alpha=0.4, hatch='///', edgecolor='white', linewidth=1.5)
    
    ax2.set_xlabel('Class', fontsize=13, fontweight='bold')
    ax2.set_ylabel('Intra-class Distance', fontsize=12, fontweight='bold', color='black')
    ax2_twin.set_ylabel('Inter-class Distance', fontsize=12, fontweight='bold', color='black')
    ax2.set_title('Class-wise Distance Comparison (Side-by-side)', fontsize=14, fontweight='bold', pad=15)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(plot_labels, rotation=45, ha='right', fontsize=10)
    
    # Combine legends
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_twin.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='best', fontsize=10, 
              framealpha=0.9, fancybox=True, shadow=True)
    
    ax2.grid(True, linestyle='--', alpha=0.3, color=COLORS['grid'], axis='y')
    ax2.spines['top'].set_visible(False)
    ax2_twin.spines['top'].set_visible(False)
    ax2.spines['left'].set_linewidth(1.5)
    ax2.spines['bottom'].set_linewidth(1.5)
    ax2_twin.spines['right'].set_linewidth(1.5)
    
    # Add main title
    fig.suptitle(f'Class-wise Distance Analysis ({source_size} → {target_size})', 
                fontsize=16, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    # Print statistics
    print(f"\nClass-wise Distance Statistics:")
    print(f"{'Class':<15} {'Pre-transfer':<25} {'Post-transfer':<25} {'Change':<20}")
    print(f"{'':<15} {'Intra':<12} {'Inter':<12} {'Intra':<12} {'Inter':<12} {'Intra':<9} {'Inter':<9}")
    print("-" * 100)
    
    for i, cls in enumerate(unique_classes):
        intra_change = post_intra[i] - pre_intra[i]
        inter_change = post_inter[i] - pre_inter[i]
        intra_pct = (intra_change / pre_intra[i] * 100) if pre_intra[i] > 0 else 0
        inter_pct = (inter_change / pre_inter[i] * 100) if pre_inter[i] > 0 else 0
        
        print(f"{plot_labels[i]:<15} {pre_intra[i]:>10.3f} {pre_inter[i]:>10.3f}   "
              f"{post_intra[i]:>10.3f} {post_inter[i]:>10.3f}   "
              f"{intra_change:>7.3f} ({intra_pct:>5.1f}%) {inter_change:>7.3f} ({inter_pct:>5.1f}%)")
    
    print(f"\nPlot saved to: {output_file}")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Class-wise Distance Scatter Plot Visualization')
    parser.add_argument('--features_dir', type=str, default='features',
                       help='Feature files directory')
    parser.add_argument('--output_dir', type=str, default='class_distance_plots',
                       help='Output directory')
    parser.add_argument('--class_labels', type=str, nargs='+', default=None,
                       help='Optional class label names (e.g., OK crimped one_missing two_missing)')
    
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
    print("Class-wise Distance Scatter Plot Visualization")
    print("=" * 60)
    print(f"Features directory: {features_dir}")
    print(f"Output directory: {output_dir}")
    print("=" * 60)
    
    try:
        # Direction 1: 0.5 → 0.35
        print("\n" + "="*60)
        print("Generating plot 1: 0.5 → 0.35")
        print("="*60)
        base_mlp_file_1 = features_dir / 'base_mlp_features_05_to_035.pkl'
        v3_file = features_dir / 'v3_features_05_to_035.pkl'
        
        if base_mlp_file_1.exists() and v3_file.exists():
            base_data = load_features(base_mlp_file_1)
            adapt_data = load_features(v3_file)
            
            plot_class_distance_scatter(
                base_data, adapt_data,
                '0.5', '0.35',
                output_dir / 'Class_Distance_Scatter_05_to_035.png',
                class_labels=args.class_labels
            )
        else:
            print(f"Warning: Feature files missing for 0.5 → 0.35")
        
        # Direction 2: 0.35 → 0.5
        print("\n" + "="*60)
        print("Generating plot 2: 0.35 → 0.5")
        print("="*60)
        base_mlp_file_2 = features_dir / 'base_mlp_features_035_to_05.pkl'
        v7_file = features_dir / 'v7_features_035_to_05.pkl'
        
        if base_mlp_file_2.exists() and v7_file.exists():
            base_data = load_features(base_mlp_file_2)
            adapt_data = load_features(v7_file)
            
            plot_class_distance_scatter(
                base_data, adapt_data,
                '0.35', '0.5',
                output_dir / 'Class_Distance_Scatter_035_to_05.png',
                class_labels=args.class_labels
            )
        else:
            print(f"Warning: Feature files missing for 0.35 → 0.5")
        
        print(f"\n{'='*60}")
        print("Class Distance Scatter Plot Visualization Complete!")
        print(f"Plots saved in: {output_dir}")
        print(f"{'='*60}")
        
    except Exception as e:
        print(f"\nVisualization failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

