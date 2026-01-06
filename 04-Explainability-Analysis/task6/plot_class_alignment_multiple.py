#!/usr/bin/env python3
"""
Multiple Visualization Methods for Class-wise Feature Alignment Comparison
顶刊常用的多种类别特征对齐效果可视化方法
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
from scipy.spatial.distance import pdist, squareform, cdist
import seaborn as sns
from matplotlib.patches import Circle, Rectangle
import matplotlib.patches as mpatches

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
CLASS_COLORS_DARK = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']


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


def compute_class_centroids(features_2d, labels):
    """Compute centroids for each class"""
    unique_classes = np.unique(labels)
    centroids = {}
    for cls in unique_classes:
        mask = labels == cls
        centroids[cls] = np.mean(features_2d[mask], axis=0)
    return centroids


def compute_class_distance_matrix(features_2d, labels):
    """Compute pairwise distance matrix between class centroids"""
    unique_classes = np.unique(labels)
    n_classes = len(unique_classes)
    centroids = compute_class_centroids(features_2d, labels)
    
    distance_matrix = np.zeros((n_classes, n_classes))
    for i, cls1 in enumerate(unique_classes):
        for j, cls2 in enumerate(unique_classes):
            if i == j:
                # Intra-class distance (mean pairwise distance within class)
                mask = labels == cls1
                class_samples = features_2d[mask]
                if len(class_samples) > 1:
                    pairwise_dist = pdist(class_samples, metric='euclidean')
                    distance_matrix[i, j] = np.mean(pairwise_dist)
                else:
                    distance_matrix[i, j] = 0
            else:
                # Inter-class distance (distance between centroids)
                distance_matrix[i, j] = np.linalg.norm(centroids[cls1] - centroids[cls2])
    
    return distance_matrix, unique_classes


def method1_heatmap_comparison(base_data, adapt_data, source_size, target_size, output_dir):
    """
    Method 1: Heatmap Comparison
    Show class distance matrices before and after transfer
    """
    print("\nGenerating Method 1: Heatmap Comparison...")
    
    # Combine features
    pre_features = np.concatenate([base_data['source_features'], base_data['target_features']], axis=0)
    pre_labels = np.concatenate([base_data['source_labels'], base_data['target_labels']], axis=0)
    post_features = np.concatenate([adapt_data['source_features'], adapt_data['target_features']], axis=0)
    post_labels = np.concatenate([adapt_data['source_labels'], adapt_data['target_labels']], axis=0)
    
    # Compute 2D coordinates
    pre_2d = compute_t_sne_2d(pre_features, random_state=42)
    post_2d = compute_t_sne_2d(post_features, random_state=42)
    
    # Compute distance matrices
    pre_dist_matrix, unique_classes = compute_class_distance_matrix(pre_2d, pre_labels)
    post_dist_matrix, _ = compute_class_distance_matrix(post_2d, post_labels)
    
    # Create figure
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
    fig.patch.set_facecolor('white')
    
    class_labels = [f'Class {int(c)}' for c in unique_classes]
    
    # Pre-transfer heatmap
    sns.heatmap(pre_dist_matrix, annot=True, fmt='.1f', cmap='YlOrRd', 
                xticklabels=class_labels, yticklabels=class_labels,
                cbar_kws={'label': 'Distance'}, ax=ax1, square=True)
    ax1.set_title('Pre-transfer\nClass Distance Matrix', fontsize=12, fontweight='bold')
    
    # Post-transfer heatmap
    sns.heatmap(post_dist_matrix, annot=True, fmt='.1f', cmap='YlOrRd',
                xticklabels=class_labels, yticklabels=class_labels,
                cbar_kws={'label': 'Distance'}, ax=ax2, square=True)
    ax2.set_title('Post-transfer\nClass Distance Matrix', fontsize=12, fontweight='bold')
    
    # Difference heatmap
    diff_matrix = post_dist_matrix - pre_dist_matrix
    sns.heatmap(diff_matrix, annot=True, fmt='.1f', cmap='RdBu_r', center=0,
                xticklabels=class_labels, yticklabels=class_labels,
                cbar_kws={'label': 'Distance Change'}, ax=ax3, square=True)
    ax3.set_title('Difference\n(Post - Pre)', fontsize=12, fontweight='bold')
    
    fig.suptitle(f'Heatmap Comparison: Class Distance Matrices ({source_size} → {target_size})',
                fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    output_file = output_dir / f'Method1_Heatmap_{source_size}_to_{target_size}.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {output_file}")


def method2_radar_chart(base_data, adapt_data, source_size, target_size, output_dir):
    """
    Method 2: Radar Chart (Spider Plot)
    Show multi-dimensional metrics for each class
    """
    print("\nGenerating Method 2: Radar Chart...")
    
    # Combine features
    pre_features = np.concatenate([base_data['source_features'], base_data['target_features']], axis=0)
    pre_labels = np.concatenate([base_data['source_labels'], base_data['target_labels']], axis=0)
    post_features = np.concatenate([adapt_data['source_features'], adapt_data['target_features']], axis=0)
    post_labels = np.concatenate([adapt_data['source_labels'], adapt_data['target_labels']], axis=0)
    
    pre_2d = compute_t_sne_2d(pre_features, random_state=42)
    post_2d = compute_t_sne_2d(post_features, random_state=42)
    
    unique_classes = np.unique(pre_labels)
    n_classes = len(unique_classes)
    
    # Compute metrics for each class
    metrics = ['Intra-Dist', 'Inter-Dist', 'Compactness', 'Separation']
    n_metrics = len(metrics)
    
    angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
    angles += angles[:1]  # Complete the circle
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12), subplot_kw=dict(projection='polar'))
    axes = axes.flatten()
    fig.patch.set_facecolor('white')
    
    for idx, cls in enumerate(unique_classes[:5]):  # Up to 5 classes
        ax = axes[idx]
        
        # Pre-transfer metrics
        pre_mask = pre_labels == cls
        pre_class_samples = pre_2d[pre_mask]
        pre_other_mask = pre_labels != cls
        pre_other_samples = pre_2d[pre_other_mask]
        
        pre_intra = np.mean(pdist(pre_class_samples)) if len(pre_class_samples) > 1 else 0
        pre_inter = np.mean(cdist(pre_class_samples, pre_other_samples)) if len(pre_other_samples) > 0 else 0
        pre_compact = 1.0 / (1.0 + pre_intra)  # Compactness (inverse of intra)
        pre_sep = pre_inter  # Separation
        
        # Post-transfer metrics
        post_mask = post_labels == cls
        post_class_samples = post_2d[post_mask]
        post_other_mask = post_labels != cls
        post_other_samples = post_2d[post_other_mask]
        
        post_intra = np.mean(pdist(post_class_samples)) if len(post_class_samples) > 1 else 0
        post_inter = np.mean(cdist(post_class_samples, post_other_samples)) if len(post_other_samples) > 0 else 0
        post_compact = 1.0 / (1.0 + post_intra)
        post_sep = post_inter
        
        # Normalize values for visualization
        max_val = max(pre_intra, pre_inter, post_intra, post_inter, pre_compact*100, post_compact*100)
        
        pre_values = [pre_intra/max_val, pre_inter/max_val, pre_compact*100/max_val, pre_sep/max_val]
        post_values = [post_intra/max_val, post_inter/max_val, post_compact*100/max_val, post_sep/max_val]
        pre_values += pre_values[:1]
        post_values += post_values[:1]
        
        ax.plot(angles, pre_values, 'o-', linewidth=2, label='Pre-transfer', color=COLORS['pre'])
        ax.fill(angles, pre_values, alpha=0.25, color=COLORS['pre'])
        ax.plot(angles, post_values, 's-', linewidth=2, label='Post-transfer', color=COLORS['post'])
        ax.fill(angles, post_values, alpha=0.25, color=COLORS['post'])
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics, fontsize=9)
        ax.set_title(f'Class {int(cls)}', fontsize=11, fontweight='bold', pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=8)
        ax.grid(True)
    
    # Hide extra subplots
    if n_classes < 5:
        for idx in range(n_classes, 5):
            axes[idx].set_visible(False)
    
    fig.suptitle(f'Radar Chart: Multi-dimensional Class Metrics ({source_size} → {target_size})',
                fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    output_file = output_dir / f'Method2_RadarChart_{source_size}_to_{target_size}.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {output_file}")


def method3_boxplot_comparison(base_data, adapt_data, source_size, target_size, output_dir):
    """
    Method 3: Box Plot Comparison
    Show distribution of intra-class distances for each class
    """
    print("\nGenerating Method 3: Box Plot Comparison...")
    
    pre_features = np.concatenate([base_data['source_features'], base_data['target_features']], axis=0)
    pre_labels = np.concatenate([base_data['source_labels'], base_data['target_labels']], axis=0)
    post_features = np.concatenate([adapt_data['source_features'], adapt_data['target_features']], axis=0)
    post_labels = np.concatenate([adapt_data['source_labels'], adapt_data['target_labels']], axis=0)
    
    pre_2d = compute_t_sne_2d(pre_features, random_state=42)
    post_2d = compute_t_sne_2d(post_features, random_state=42)
    
    unique_classes = np.unique(pre_labels)
    
    # Compute intra-class distances for each sample
    pre_intra_distances = []
    post_intra_distances = []
    class_list = []
    
    for cls in unique_classes:
        pre_mask = pre_labels == cls
        pre_class_samples = pre_2d[pre_mask]
        if len(pre_class_samples) > 1:
            pre_dist = pdist(pre_class_samples, metric='euclidean')
            pre_intra_distances.extend(pre_dist)
            class_list.extend([f'Class {int(cls)}'] * len(pre_dist))
        
        post_mask = post_labels == cls
        post_class_samples = post_2d[post_mask]
        if len(post_class_samples) > 1:
            post_dist = pdist(post_class_samples, metric='euclidean')
            post_intra_distances.extend(post_dist)
    
    # Prepare data for plotting
    pre_data_list = []
    post_data_list = []
    class_labels_plot = []
    
    for cls in unique_classes:
        pre_mask = pre_labels == cls
        pre_class_samples = pre_2d[pre_mask]
        if len(pre_class_samples) > 1:
            # Compute distance from each sample to centroid
            centroid = np.mean(pre_class_samples, axis=0)
            distances = np.linalg.norm(pre_class_samples - centroid, axis=1)
            pre_data_list.append(distances)
        
        post_mask = post_labels == cls
        post_class_samples = post_2d[post_mask]
        if len(post_class_samples) > 1:
            centroid = np.mean(post_class_samples, axis=0)
            distances = np.linalg.norm(post_class_samples - centroid, axis=1)
            post_data_list.append(distances)
        
        class_labels_plot.append(f'Class {int(cls)}')
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.patch.set_facecolor('white')
    
    # Pre-transfer boxplot
    bp1 = ax1.boxplot(pre_data_list, tick_labels=class_labels_plot, patch_artist=True,
                     showmeans=True, meanline=True)
    for patch in bp1['boxes']:
        patch.set_facecolor(COLORS['pre'])
        patch.set_alpha(0.7)
    ax1.set_ylabel('Distance to Class Centroid', fontsize=12, fontweight='bold')
    ax1.set_title('Pre-transfer\nIntra-class Distance Distribution', fontsize=12, fontweight='bold')
    ax1.grid(True, linestyle='--', alpha=0.3, axis='y')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    # Post-transfer boxplot
    bp2 = ax2.boxplot(post_data_list, tick_labels=class_labels_plot, patch_artist=True,
                     showmeans=True, meanline=True)
    for patch in bp2['boxes']:
        patch.set_facecolor(COLORS['post'])
        patch.set_alpha(0.7)
    ax2.set_ylabel('Distance to Class Centroid', fontsize=12, fontweight='bold')
    ax2.set_title('Post-transfer\nIntra-class Distance Distribution', fontsize=12, fontweight='bold')
    ax2.grid(True, linestyle='--', alpha=0.3, axis='y')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    
    fig.suptitle(f'Box Plot Comparison: Intra-class Distance Distribution ({source_size} → {target_size})',
                fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    output_file = output_dir / f'Method3_BoxPlot_{source_size}_to_{target_size}.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {output_file}")


def method4_centroid_trajectory(base_data, adapt_data, source_size, target_size, output_dir):
    """
    Method 4: Centroid Trajectory Plot
    Show how class centroids move in feature space
    """
    print("\nGenerating Method 4: Centroid Trajectory...")
    
    pre_features = np.concatenate([base_data['source_features'], base_data['target_features']], axis=0)
    pre_labels = np.concatenate([base_data['source_labels'], base_data['target_labels']], axis=0)
    post_features = np.concatenate([adapt_data['source_features'], adapt_data['target_features']], axis=0)
    post_labels = np.concatenate([adapt_data['source_labels'], adapt_data['target_labels']], axis=0)
    
    # Use same t-SNE initialization for alignment
    combined_features = np.concatenate([pre_features, post_features], axis=0)
    combined_2d = compute_t_sne_2d(combined_features, random_state=42)
    
    n_pre = len(pre_features)
    pre_2d = combined_2d[:n_pre]
    post_2d = combined_2d[n_pre:]
    
    unique_classes = np.unique(pre_labels)
    pre_centroids = compute_class_centroids(pre_2d, pre_labels)
    post_centroids = compute_class_centroids(post_2d, post_labels)
    
    fig, ax = plt.subplots(figsize=(12, 10))
    fig.patch.set_facecolor('white')
    ax.set_facecolor(COLORS['background'])
    
    # Plot pre-transfer centroids
    for i, cls in enumerate(unique_classes):
        x_pre, y_pre = pre_centroids[cls]
        x_post, y_post = post_centroids[cls]
        
        # Draw trajectory arrow
        ax.annotate('', xy=(x_post, y_post), xytext=(x_pre, y_pre),
                   arrowprops=dict(arrowstyle='->', lw=3, color=CLASS_COLORS_DARK[i % len(CLASS_COLORS_DARK)],
                                 alpha=0.7, zorder=1))
        
        # Plot centroids
        ax.scatter(x_pre, y_pre, s=300, c=COLORS['pre'], marker='o', 
                  edgecolors='white', linewidths=2, alpha=0.8, zorder=3,
                  label=f'Pre Class {int(cls)}' if i == 0 else '')
        ax.scatter(x_post, y_post, s=300, c=COLORS['post'], marker='s',
                  edgecolors='white', linewidths=2, alpha=0.8, zorder=3,
                  label=f'Post Class {int(cls)}' if i == 0 else '')
        
        # Add class labels
        ax.text(x_pre, y_pre, f' {int(cls)}', fontsize=11, fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    ax.set_xlabel('t-SNE Dimension 1', fontsize=12, fontweight='bold')
    ax.set_ylabel('t-SNE Dimension 2', fontsize=12, fontweight='bold')
    ax.set_title(f'Centroid Trajectory: Class Movement in Feature Space ({source_size} → {target_size})',
                fontsize=13, fontweight='bold', pad=15)
    ax.legend(loc='best', fontsize=10, framealpha=0.9)
    ax.grid(True, linestyle='--', alpha=0.3, color=COLORS['grid'])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    
    output_file = output_dir / f'Method4_CentroidTrajectory_{source_size}_to_{target_size}.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {output_file}")


def method5_improved_scatter(base_data, adapt_data, source_size, target_size, output_dir):
    """
    Method 5: Improved Scatter Plot with Confidence Ellipses
    Show class distributions with confidence regions
    """
    print("\nGenerating Method 5: Improved Scatter with Confidence Ellipses...")
    
    from matplotlib.patches import Ellipse
    from scipy.stats import chi2
    
    pre_features = np.concatenate([base_data['source_features'], base_data['target_features']], axis=0)
    pre_labels = np.concatenate([base_data['source_labels'], base_data['target_labels']], axis=0)
    post_features = np.concatenate([adapt_data['source_features'], adapt_data['target_features']], axis=0)
    post_labels = np.concatenate([adapt_data['source_labels'], adapt_data['target_labels']], axis=0)
    
    pre_2d = compute_t_sne_2d(pre_features, random_state=42)
    post_2d = compute_t_sne_2d(post_features, random_state=42)
    
    unique_classes = np.unique(pre_labels)
    
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
    
    ax1.set_title('Pre-transfer\n(with 95% Confidence Ellipses)', fontsize=12, fontweight='bold')
    ax1.legend(loc='best', fontsize=9, framealpha=0.9)
    ax2.set_title('Post-transfer\n(with 95% Confidence Ellipses)', fontsize=12, fontweight='bold')
    
    fig.suptitle(f'Improved Scatter: Class Distributions with Confidence Regions ({source_size} → {target_size})',
                fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    output_file = output_dir / f'Method5_ImprovedScatter_{source_size}_to_{target_size}.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {output_file}")


def method6_metrics_comparison(base_data, adapt_data, source_size, target_size, output_dir):
    """
    Method 6: Comprehensive Metrics Comparison
    Multiple metrics in a clean comparison format
    """
    print("\nGenerating Method 6: Comprehensive Metrics Comparison...")
    
    pre_features = np.concatenate([base_data['source_features'], base_data['target_features']], axis=0)
    pre_labels = np.concatenate([base_data['source_labels'], base_data['target_labels']], axis=0)
    post_features = np.concatenate([adapt_data['source_features'], adapt_data['target_features']], axis=0)
    post_labels = np.concatenate([adapt_data['source_labels'], adapt_data['target_labels']], axis=0)
    
    pre_2d = compute_t_sne_2d(pre_features, random_state=42)
    post_2d = compute_t_sne_2d(post_features, random_state=42)
    
    unique_classes = np.unique(pre_labels)
    n_classes = len(unique_classes)
    
    # Compute metrics
    metrics_pre = {'intra': [], 'inter': [], 'silhouette': []}
    metrics_post = {'intra': [], 'inter': [], 'silhouette': []}
    
    for cls in unique_classes:
        # Pre-transfer
        pre_mask = pre_labels == cls
        pre_class_samples = pre_2d[pre_mask]
        pre_other_mask = pre_labels != cls
        pre_other_samples = pre_2d[pre_other_mask]
        
        pre_intra = np.mean(pdist(pre_class_samples)) if len(pre_class_samples) > 1 else 0
        pre_inter = np.mean(cdist(pre_class_samples, pre_other_samples)) if len(pre_other_samples) > 0 else 0
        pre_sil = (pre_inter - pre_intra) / max(pre_inter, pre_intra) if max(pre_inter, pre_intra) > 0 else 0
        
        metrics_pre['intra'].append(pre_intra)
        metrics_pre['inter'].append(pre_inter)
        metrics_pre['silhouette'].append(pre_sil)
        
        # Post-transfer
        post_mask = post_labels == cls
        post_class_samples = post_2d[post_mask]
        post_other_mask = post_labels != cls
        post_other_samples = post_2d[post_other_mask]
        
        post_intra = np.mean(pdist(post_class_samples)) if len(post_class_samples) > 1 else 0
        post_inter = np.mean(cdist(post_class_samples, post_other_samples)) if len(post_other_samples) > 0 else 0
        post_sil = (post_inter - post_intra) / max(post_inter, post_intra) if max(post_inter, post_intra) > 0 else 0
        
        metrics_post['intra'].append(post_intra)
        metrics_post['inter'].append(post_inter)
        metrics_post['silhouette'].append(post_sil)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.patch.set_facecolor('white')
    axes = axes.flatten()
    
    class_labels = [f'Class {int(c)}' for c in unique_classes]
    x_pos = np.arange(n_classes)
    width = 0.35
    
    # Intra-class distance
    ax = axes[0]
    bars1 = ax.bar(x_pos - width/2, metrics_pre['intra'], width, label='Pre-transfer',
                  color=COLORS['pre'], alpha=0.7, edgecolor='white', linewidth=1.5)
    bars2 = ax.bar(x_pos + width/2, metrics_post['intra'], width, label='Post-transfer',
                  color=COLORS['post'], alpha=0.7, edgecolor='white', linewidth=1.5)
    ax.set_ylabel('Intra-class Distance', fontsize=12, fontweight='bold')
    ax.set_title('Intra-class Distance (Lower is Better)', fontsize=12, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(class_labels)
    ax.legend(fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.3, axis='y')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Inter-class distance
    ax = axes[1]
    bars1 = ax.bar(x_pos - width/2, metrics_pre['inter'], width, label='Pre-transfer',
                  color=COLORS['pre'], alpha=0.7, edgecolor='white', linewidth=1.5)
    bars2 = ax.bar(x_pos + width/2, metrics_post['inter'], width, label='Post-transfer',
                  color=COLORS['post'], alpha=0.7, edgecolor='white', linewidth=1.5)
    ax.set_ylabel('Inter-class Distance', fontsize=12, fontweight='bold')
    ax.set_title('Inter-class Distance (Higher is Better)', fontsize=12, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(class_labels)
    ax.legend(fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.3, axis='y')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Separation ratio
    ax = axes[2]
    pre_ratio = [m_inter / m_intra if m_intra > 0 else 0 
                for m_intra, m_inter in zip(metrics_pre['intra'], metrics_pre['inter'])]
    post_ratio = [m_inter / m_intra if m_intra > 0 else 0 
                 for m_intra, m_inter in zip(metrics_post['intra'], metrics_post['inter'])]
    bars1 = ax.bar(x_pos - width/2, pre_ratio, width, label='Pre-transfer',
                  color=COLORS['pre'], alpha=0.7, edgecolor='white', linewidth=1.5)
    bars2 = ax.bar(x_pos + width/2, post_ratio, width, label='Post-transfer',
                  color=COLORS['post'], alpha=0.7, edgecolor='white', linewidth=1.5)
    ax.set_ylabel('Separation Ratio (Inter/Intra)', fontsize=12, fontweight='bold')
    ax.set_title('Separation Ratio (Higher is Better)', fontsize=12, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(class_labels)
    ax.legend(fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.3, axis='y')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Improvement percentage
    ax = axes[3]
    intra_improvement = [((pre - post) / pre * 100) if pre > 0 else 0 
                        for pre, post in zip(metrics_pre['intra'], metrics_post['intra'])]
    inter_improvement = [((post - pre) / pre * 100) if pre > 0 else 0 
                        for pre, post in zip(metrics_pre['inter'], metrics_post['inter'])]
    
    x = np.arange(n_classes)
    bars1 = ax.bar(x - width/2, intra_improvement, width, label='Intra-class Reduction %',
                  color='#66C2A5', alpha=0.7, edgecolor='white', linewidth=1.5)
    bars2 = ax.bar(x + width/2, inter_improvement, width, label='Inter-class Increase %',
                  color='#FC8D62', alpha=0.7, edgecolor='white', linewidth=1.5)
    ax.set_ylabel('Improvement (%)', fontsize=12, fontweight='bold')
    ax.set_title('Improvement Percentage', fontsize=12, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(class_labels)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    ax.legend(fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.3, axis='y')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    fig.suptitle(f'Comprehensive Metrics Comparison ({source_size} → {target_size})',
                fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    output_file = output_dir / f'Method6_ComprehensiveMetrics_{source_size}_to_{target_size}.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {output_file}")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Multiple Visualization Methods for Class Alignment')
    parser.add_argument('--features_dir', type=str, default='features',
                       help='Feature files directory')
    parser.add_argument('--output_dir', type=str, default='class_alignment_methods',
                       help='Output directory')
    parser.add_argument('--methods', type=str, nargs='+', 
                       choices=['1', '2', '3', '4', '5', '6', 'all'],
                       default='all',
                       help='Which visualization methods to generate (1-6 or all)')
    parser.add_argument('--direction', type=str, choices=['05_to_035', '035_to_05', 'both'],
                       default='both',
                       help='Which transfer direction(s) to visualize')
    
    args = parser.parse_args()
    
    base_dir = Path(__file__).parent
    features_dir = Path(args.features_dir)
    if not features_dir.is_absolute():
        features_dir = base_dir / features_dir
    
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = base_dir / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    methods_to_run = args.methods if 'all' not in args.methods else ['1', '2', '3', '4', '5', '6']
    
    print("=" * 60)
    print("Multiple Visualization Methods for Class Alignment")
    print("=" * 60)
    print(f"Features directory: {features_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Methods: {', '.join(methods_to_run)}")
    print(f"Direction(s): {args.direction}")
    print("=" * 60)
    
    methods_dict = {
        '1': method1_heatmap_comparison,
        '2': method2_radar_chart,
        '3': method3_boxplot_comparison,
        '4': method4_centroid_trajectory,
        '5': method5_improved_scatter,
        '6': method6_metrics_comparison
    }
    
    method_names = {
        '1': 'Heatmap Comparison',
        '2': 'Radar Chart',
        '3': 'Box Plot Comparison',
        '4': 'Centroid Trajectory',
        '5': 'Improved Scatter with Confidence Ellipses',
        '6': 'Comprehensive Metrics Comparison'
    }
    
    directions_to_process = []
    if args.direction in ['05_to_035', 'both']:
        directions_to_process.append(('05_to_035', '0.5', '0.35'))
    if args.direction in ['035_to_05', 'both']:
        directions_to_process.append(('035_to_05', '0.35', '0.5'))
    
    try:
        for direction_key, source_size, target_size in directions_to_process:
            print(f"\n{'='*60}")
            print(f"Processing direction: {source_size} → {target_size}")
            print(f"{'='*60}")
            
            # Convert sizes to match file naming (0.5 -> 05, 0.35 -> 035)
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
            
            for method_num in methods_to_run:
                method_func = methods_dict[method_num]
                method_name = method_names[method_num]
                print(f"\nMethod {method_num}: {method_name}")
                try:
                    method_func(base_data, adapt_data, source_size, target_size, output_dir)
                except Exception as e:
                    print(f"  Error in method {method_num}: {e}")
                    import traceback
                    traceback.print_exc()
        
        print(f"\n{'='*60}")
        print("All visualizations complete!")
        print(f"Results saved in: {output_dir}")
        print(f"{'='*60}")
        print("\nAvailable visualization methods:")
        for num, name in method_names.items():
            print(f"  Method {num}: {name}")
        
    except Exception as e:
        print(f"\nVisualization failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

