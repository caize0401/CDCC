#!/usr/bin/env python3
"""
Improved Scatter Plot with Confidence Ellipses
Visualize class distributions with 95% confidence ellipses for feature alignment comparison
Based on Method 5 from plot_class_alignment_multiple.py
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
from matplotlib.patches import Ellipse

# Set font and plotting style (journal quality)
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

# Journal-quality color scheme
COLORS = {
    'pre': '#2E86AB',       # Deep blue - Pre-transfer
    'post': '#A23B72',      # Purple-red - Post-transfer
    'background': '#F5F5F5', # Light gray background
    'grid': '#E0E0E0'        # Grid line color
}

# Color palette for classes
CLASS_COLORS = ['#66C2A5', '#FC8D62', '#8DA0CB', '#E78AC3', '#A6D854']


def load_features(feature_file: Path):
    """Load feature file"""
    with open(feature_file, 'rb') as f:
        data = pickle.load(f)
    return data


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


def plot_confidence_ellipses(base_data, adapt_data, source_size, target_size, output_file):
    """
    Method 5: Improved Scatter Plot with Confidence Ellipses
    Show class distributions with confidence regions
    (Original code from plot_class_alignment_multiple.py, no modifications)
    """
    print(f"\n{'='*60}")
    print(f"Generating Confidence Ellipses Plot: {source_size} → {target_size}")
    print(f"{'='*60}")
    
    pre_features = np.concatenate([base_data['source_features'], base_data['target_features']], axis=0)
    pre_labels = np.concatenate([base_data['source_labels'], base_data['target_labels']], axis=0)
    post_features = np.concatenate([adapt_data['source_features'], adapt_data['target_features']], axis=0)
    post_labels = np.concatenate([adapt_data['source_labels'], adapt_data['target_labels']], axis=0)
    
    pre_2d = compute_t_sne_2d(pre_features, random_state=42)
    post_2d = compute_t_sne_2d(post_features, random_state=42)
    
    unique_classes = np.unique(pre_labels)
    
    print(f"Classes found: {[int(c) for c in unique_classes]}")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    fig.patch.set_facecolor('white')
    
    for idx, cls in enumerate(unique_classes):
        color = CLASS_COLORS[idx % len(CLASS_COLORS)]
        
        # Pre-transfer
        pre_mask = pre_labels == cls
        pre_class_samples = pre_2d[pre_mask]
        if len(pre_class_samples) > 2:
            centroid = np.mean(pre_class_samples, axis=0)
            cov = np.cov(pre_class_samples.T)
            
            # Draw confidence ellipse (2 standard deviations)
            eigenvals, eigenvecs = np.linalg.eigh(cov)
            order = eigenvals.argsort()[::-1]
            eigenvals, eigenvecs = eigenvals[order], eigenvecs[:, order]
            angle = np.degrees(np.arctan2(*eigenvecs[:, 0][::-1]))
            width, height = 2 * np.sqrt(eigenvals) * 2  # 2 std devs
            
            ellipse = Ellipse(xy=centroid, width=width, height=height, angle=angle,
                            facecolor=color, alpha=0.2, edgecolor=color, linewidth=2)
            ax1.add_patch(ellipse)
            
            ax1.scatter(pre_class_samples[:, 0], pre_class_samples[:, 1], 
                       c=[color], s=30, alpha=0.5, marker='o', edgecolors='none')
            ax1.scatter(centroid[0], centroid[1], c=[color], s=150, marker='*', 
                       edgecolors='white', linewidths=1.5, zorder=5,
                       label=f'Class {int(cls)}' if idx < 5 else '')
        
        # Post-transfer
        post_mask = post_labels == cls
        post_class_samples = post_2d[post_mask]
        if len(post_class_samples) > 2:
            centroid = np.mean(post_class_samples, axis=0)
            cov = np.cov(post_class_samples.T)
            
            eigenvals, eigenvecs = np.linalg.eigh(cov)
            order = eigenvals.argsort()[::-1]
            eigenvals, eigenvecs = eigenvals[order], eigenvecs[:, order]
            angle = np.degrees(np.arctan2(*eigenvecs[:, 0][::-1]))
            width, height = 2 * np.sqrt(eigenvals) * 2
            
            ellipse = Ellipse(xy=centroid, width=width, height=height, angle=angle,
                            facecolor=color, alpha=0.2, edgecolor=color, linewidth=2, linestyle='--')
            ax2.add_patch(ellipse)
            
            ax2.scatter(post_class_samples[:, 0], post_class_samples[:, 1],
                       c=[color], s=30, alpha=0.5, marker='s', edgecolors='none')
            ax2.scatter(centroid[0], centroid[1], c=[color], s=150, marker='*',
                       edgecolors='white', linewidths=1.5, zorder=5)
    
    for ax in [ax1, ax2]:
        ax.set_xlabel('t-SNE Dimension 1', fontsize=12, fontweight='bold')
        ax.set_ylabel('t-SNE Dimension 2', fontsize=12, fontweight='bold')
        ax.grid(True, linestyle='--', alpha=0.3, color=COLORS['grid'])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_facecolor(COLORS['background'])
    
    ax1.set_title('Pre-transfer (95% Confidence Ellipses)', fontsize=12, fontweight='bold')
    ax1.legend(loc='best', fontsize=9, framealpha=0.9)
    ax2.set_title('Post-transfer (95% Confidence Ellipses)', fontsize=12, fontweight='bold')
    
    fig.suptitle(f'Scatter Plot of Class Distributions with Confidence Ellipses ({source_size} → {target_size})',
                fontsize=14, fontweight='bold', y=1.00)
    plt.tight_layout()
    
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"Plot saved to: {output_file}")
    
    # Print summary
    print(f"\nClass Distribution Summary:")
    print(f"{'Class':<10} {'Pre-transfer':<15} {'Post-transfer':<15}")
    print(f"{'':<10} {'Samples':<15} {'Samples':<15}")
    print("-" * 50)
    
    for cls in unique_classes:
        pre_count = np.sum(pre_labels == cls)
        post_count = np.sum(post_labels == cls)
        print(f"Class {int(cls):<7} {pre_count:>13} {post_count:>13}")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Confidence Ellipses Visualization for Class Alignment')
    parser.add_argument('--features_dir', type=str, default='features',
                       help='Feature files directory')
    parser.add_argument('--output_dir', type=str, default='confidence_ellipses',
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
    print("Confidence Ellipses Visualization")
    print("(Original Method 5 - All samples per class, no domain distinction)")
    print("=" * 60)
    print(f"Features directory: {features_dir}")
    print(f"Output directory: {output_dir}")
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
            
            output_file = output_dir / f'Confidence_Ellipses_{source_size}_to_{target_size}.png'
            plot_confidence_ellipses(base_data, adapt_data, source_size, target_size, output_file)
        
        print(f"\n{'='*60}")
        print("Confidence Ellipses Visualization Complete!")
        print(f"Results saved in: {output_dir}")
        print(f"{'='*60}")
        
    except Exception as e:
        print(f"\nVisualization failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
