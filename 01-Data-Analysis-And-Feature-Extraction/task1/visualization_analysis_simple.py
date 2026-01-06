import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# 设置matplotlib后端为非交互式
import matplotlib
matplotlib.use('Agg')

# 设置中文字体和样式
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def load_feature_data():
    """加载特征数据"""
    print("加载特征数据...")
    features_035 = pd.read_pickle("task1/features_035.pkl")
    features_05 = pd.read_pickle("task1/features_05.pkl")
    
    print(f"0.35数据集特征形状: {features_035.shape}")
    print(f"0.5数据集特征形状: {features_05.shape}")
    
    return features_035, features_05

def get_feature_columns(df):
    """获取特征列名（排除标签列）"""
    exclude_cols = ['CrimpID', 'Wire_cross-section_conductor', 'Main_label_string', 
                   'Sub_label_string', 'Main-label_encoded', 'Sub_label_encoded', 
                   'Binary_label_encoded', 'CFM_label_encoded']
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    return feature_cols

def create_fault_classification_plots(features_035, features_05):
    """创建故障分类部分的可视化图"""
    print("创建故障分类部分可视化图...")
    
    # 获取特征列
    feature_cols = get_feature_columns(features_035)
    
    # 1. 特征分布对比图（0.35数据集）
    print("创建图1: 0.35数据集特征分布对比图")
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Feature Distribution Comparison for 0.35 Dataset', fontsize=16, fontweight='bold')
    
    # 选择几个重要特征进行展示
    important_features = ['mean', 'std', 'skewness', 'spectral_centroid']
    
    for i, feature in enumerate(important_features):
        if feature in feature_cols:
            ax = axes[i//2, i%2]
            
            # 按故障类型分组绘制
            fault_types = features_035['Sub_label_string'].unique()
            colors = plt.cm.Set3(np.linspace(0, 1, len(fault_types)))
            
            for j, fault_type in enumerate(fault_types):
                data = features_035[features_035['Sub_label_string'] == fault_type][feature]
                ax.hist(data, alpha=0.7, label=fault_type, color=colors[j], bins=30)
            
            ax.set_title(f'{feature} Distribution')
            ax.set_xlabel(feature)
            ax.set_ylabel('Frequency')
            ax.legend()
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('task1/图1_035数据集特征分布对比图.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. 特征分布对比图（0.5数据集）
    print("创建图2: 0.5数据集特征分布对比图")
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Feature Distribution Comparison for 0.5 Dataset', fontsize=16, fontweight='bold')
    
    for i, feature in enumerate(important_features):
        if feature in feature_cols:
            ax = axes[i//2, i%2]
            
            fault_types = features_05['Sub_label_string'].unique()
            colors = plt.cm.Set3(np.linspace(0, 1, len(fault_types)))
            
            for j, fault_type in enumerate(fault_types):
                data = features_05[features_05['Sub_label_string'] == fault_type][feature]
                ax.hist(data, alpha=0.7, label=fault_type, color=colors[j], bins=30)
            
            ax.set_title(f'{feature} Distribution')
            ax.set_xlabel(feature)
            ax.set_ylabel('Frequency')
            ax.legend()
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('task1/图2_05数据集特征分布对比图.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. 箱线图（0.35数据集）
    print("创建图3: 0.35数据集箱线图")
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Box Plots for 0.35 Dataset', fontsize=16, fontweight='bold')
    
    for i, feature in enumerate(important_features):
        if feature in feature_cols:
            ax = axes[i//2, i%2]
            
            data_for_box = []
            labels_for_box = []
            
            for fault_type in features_035['Sub_label_string'].unique():
                data = features_035[features_035['Sub_label_string'] == fault_type][feature]
                data_for_box.append(data)
                labels_for_box.append(fault_type)
            
            ax.boxplot(data_for_box, labels=labels_for_box)
            ax.set_title(f'{feature} Box Plot')
            ax.set_ylabel(feature)
            ax.tick_params(axis='x', rotation=45)
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('task1/图3_035数据集箱线图.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. 箱线图（0.5数据集）
    print("创建图4: 0.5数据集箱线图")
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Box Plots for 0.5 Dataset', fontsize=16, fontweight='bold')
    
    for i, feature in enumerate(important_features):
        if feature in feature_cols:
            ax = axes[i//2, i%2]
            
            data_for_box = []
            labels_for_box = []
            
            for fault_type in features_05['Sub_label_string'].unique():
                data = features_05[features_05['Sub_label_string'] == fault_type][feature]
                data_for_box.append(data)
                labels_for_box.append(fault_type)
            
            ax.boxplot(data_for_box, labels=labels_for_box)
            ax.set_title(f'{feature} Box Plot')
            ax.set_ylabel(feature)
            ax.tick_params(axis='x', rotation=45)
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('task1/图4_05数据集箱线图.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 5. 不同数据集相同故障的特征分布对比图
    print("创建图5-9: 不同数据集相同故障的特征分布对比图")
    fault_types = ['OK', 'One Missing Strand', 'Two Missing Strands', 'Three Missing Strands', 'Crimped Insulation']
    
    for fault_type in fault_types:
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'Feature Distribution Comparison for {fault_type} Across Datasets', fontsize=16, fontweight='bold')
        
        for i, feature in enumerate(important_features):
            if feature in feature_cols:
                ax = axes[i//2, i%2]
                
                # 0.35数据集
                data_035 = features_035[features_035['Sub_label_string'] == fault_type][feature]
                # 0.5数据集
                data_05 = features_05[features_05['Sub_label_string'] == fault_type][feature]
                
                if len(data_035) > 0 and len(data_05) > 0:
                    ax.hist(data_035, alpha=0.7, label='0.35 Dataset', bins=20, color='blue')
                    ax.hist(data_05, alpha=0.7, label='0.5 Dataset', bins=20, color='red')
                    
                    ax.set_title(f'{feature} - {fault_type}')
                    ax.set_xlabel(feature)
                    ax.set_ylabel('Frequency')
                    ax.legend()
                    ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'task1/图{5 + fault_types.index(fault_type)}_{fault_type}特征分布对比图.png', dpi=300, bbox_inches='tight')
        plt.close()

def create_transfer_learning_plots(features_035, features_05):
    """创建迁移学习部分的可视化图"""
    print("创建迁移学习部分可视化图...")
    
    # 合并两个数据集
    features_combined = pd.concat([features_035, features_05], ignore_index=True)
    feature_cols = get_feature_columns(features_combined)
    
    # 1. 特征分布对比图（整体样本对比）
    print("创建图10: 整体样本特征分布对比图")
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Overall Feature Distribution Comparison Between Datasets', fontsize=16, fontweight='bold')
    
    important_features = ['mean', 'std', 'skewness', 'spectral_centroid']
    
    for i, feature in enumerate(important_features):
        if feature in feature_cols:
            ax = axes[i//2, i%2]
            
            data_035 = features_035[feature]
            data_05 = features_05[feature]
            
            ax.hist(data_035, alpha=0.7, label='0.35 Dataset', bins=30, color='blue')
            ax.hist(data_05, alpha=0.7, label='0.5 Dataset', bins=30, color='red')
            
            ax.set_title(f'{feature} Distribution')
            ax.set_xlabel(feature)
            ax.set_ylabel('Frequency')
            ax.legend()
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('task1/图10_整体样本特征分布对比图.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. 特征相关性热图
    print("创建图11: 特征相关性热图")
    # 选择数值特征进行相关性分析
    numeric_features = features_combined[feature_cols].select_dtypes(include=[np.number])
    
    # 计算相关性矩阵
    corr_matrix = numeric_features.corr()
    
    # 创建热图
    plt.figure(figsize=(12, 10))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', center=0,
                square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
    plt.title('Feature Correlation Heatmap', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('task1/图11_特征相关性热图.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. 特征箱线图（整体对比）
    print("创建图12: 整体特征箱线图")
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Overall Feature Box Plots Comparison', fontsize=16, fontweight='bold')
    
    for i, feature in enumerate(important_features):
        if feature in feature_cols:
            ax = axes[i//2, i%2]
            
            data_035 = features_035[feature]
            data_05 = features_05[feature]
            
            ax.boxplot([data_035, data_05], labels=['0.35 Dataset', '0.5 Dataset'])
            ax.set_title(f'{feature} Box Plot')
            ax.set_ylabel(feature)
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('task1/图12_整体特征箱线图.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_domain_analysis_plots(features_035, features_05):
    """创建域分析可视化图"""
    print("创建域分析可视化图...")
    
    feature_cols = get_feature_columns(features_035)
    
    # 1. 特征差异热图
    print("创建图13: 特征差异热图")
    # 计算两个数据集的特征均值差异
    mean_035 = features_035[feature_cols].mean()
    mean_05 = features_05[feature_cols].mean()
    
    # 计算相对差异
    feature_diff = np.abs(mean_035 - mean_05) / (np.abs(mean_035) + np.abs(mean_05) + 1e-10)
    
    # 创建差异热图
    plt.figure(figsize=(12, 8))
    feature_diff_sorted = feature_diff.sort_values(ascending=False)
    
    # 选择前20个差异最大的特征
    top_features = feature_diff_sorted.head(20)
    
    plt.barh(range(len(top_features)), top_features.values)
    plt.yticks(range(len(top_features)), top_features.index)
    plt.xlabel('Relative Difference')
    plt.title('Top 20 Features with Largest Differences Between Datasets', fontsize=16, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('task1/图13_特征差异热图.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. PCA域相似性分析
    print("创建图14: PCA域相似性分析")
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    
    # 准备数据
    X_035 = features_035[feature_cols].values
    X_05 = features_05[feature_cols].values
    
    # 标准化
    scaler = StandardScaler()
    X_035_scaled = scaler.fit_transform(X_035)
    X_05_scaled = scaler.transform(X_05)
    
    # PCA降维
    pca = PCA(n_components=2)
    X_035_pca = pca.fit_transform(X_035_scaled)
    X_05_pca = pca.transform(X_05_scaled)
    
    # 绘制PCA结果
    plt.figure(figsize=(12, 8))
    plt.scatter(X_035_pca[:, 0], X_035_pca[:, 1], alpha=0.6, label='0.35 Dataset', color='blue')
    plt.scatter(X_05_pca[:, 0], X_05_pca[:, 1], alpha=0.6, label='0.5 Dataset', color='red')
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
    plt.title('PCA Domain Similarity Analysis', fontsize=16, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('task1/图14_PCA域相似性分析.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. 特征重要性分析
    print("创建图15: 特征重要性分析")
    # 使用方差作为特征重要性的简单度量
    feature_importance = features_035[feature_cols].var()
    feature_importance_sorted = feature_importance.sort_values(ascending=False)
    
    # 选择前15个最重要的特征
    top_15_features = feature_importance_sorted.head(15)
    
    plt.figure(figsize=(12, 8))
    plt.barh(range(len(top_15_features)), top_15_features.values)
    plt.yticks(range(len(top_15_features)), top_15_features.index)
    plt.xlabel('Variance (Feature Importance)')
    plt.title('Top 15 Most Important Features (by Variance)', fontsize=16, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('task1/图15_特征重要性分析.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    """主函数"""
    print("开始可视化分析...")
    
    # 加载特征数据
    features_035, features_05 = load_feature_data()
    
    # 创建故障分类部分的可视化图
    create_fault_classification_plots(features_035, features_05)
    
    # 创建迁移学习部分的可视化图
    create_transfer_learning_plots(features_035, features_05)
    
    # 创建域分析可视化图
    create_domain_analysis_plots(features_035, features_05)
    
    print("所有可视化图创建完成！")

if __name__ == "__main__":
    main()

