import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体和样式
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 加载特征数据
print("加载特征数据...")
try:
    features_035 = pd.read_pickle("features_035.pkl")
    features_05 = pd.read_pickle("features_05.pkl")
except Exception as e:
    print(f"使用pandas读取失败: {e}")
    print("尝试使用joblib加载...")
    try:
        import joblib
        features_035 = joblib.load("features_035.pkl")
        features_05 = joblib.load("features_05.pkl")
    except Exception as e2:
        print(f"使用joblib读取也失败: {e2}")
        raise

# 获取数据中实际存在的故障类型（参考图1的做法）
fault_types_035 = sorted(features_035['Sub_label_string'].unique())
fault_types_05 = sorted(features_05['Sub_label_string'].unique())
# 合并两个数据集的故障类型，确保所有类型都被包含
all_fault_types = sorted(list(set(fault_types_035) | set(fault_types_05)))
print(f"检测到的故障类型: {all_fault_types}")

# 使用Set3配色方案，为所有故障类型分配颜色（参考图1的做法）
colors_palette = plt.cm.Set3(np.linspace(0, 1, len(all_fault_types)))
# 创建故障类型到颜色的映射，确保相同故障类型在所有子图中颜色一致
fault_type_colors = {fault_type: colors_palette[i] for i, fault_type in enumerate(all_fault_types)}

# 选择3个特征
selected_features = {
    'mean': 'Time Domain - Mean',
    'spectral_centroid': 'Frequency Domain - Spectral Centroid',
    'wavelet_energy_A4': 'Time-Frequency Domain - Wavelet Energy A4'
}

# 创建图形：2行3列（2个数据集 × 3个特征）
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
# 删除大标题

# 数据集列表
datasets = [
    ('0.35', features_035),
    ('0.5', features_05)
]

# 遍历数据集和特征
for dataset_idx, (dataset_name, features_df) in enumerate(datasets):
    for feature_idx, (feature_name, feature_label) in enumerate(selected_features.items()):
        ax = axes[dataset_idx, feature_idx]
        
        # 检查特征是否存在
        if feature_name not in features_df.columns:
            print(f"警告: 特征 {feature_name} 在 {dataset_name} 数据集中不存在")
            ax.text(0.5, 0.5, f'Feature {feature_name}\nnot found', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'{dataset_name} Dataset - {feature_label}', fontsize=12, fontweight='bold')
            continue
        
        # 获取该数据集中实际存在的故障类型（参考图1的做法）
        fault_types = sorted(features_df['Sub_label_string'].unique())
        
        # 按故障类型绘制直方图（参考图1的做法）
        for fault_type in fault_types:
            data = features_df[features_df['Sub_label_string'] == fault_type][feature_name]
            if len(data) > 0:
                ax.hist(data, alpha=0.7, label=fault_type, 
                       color=fault_type_colors[fault_type], bins=20)
        
        # 设置标题和标签
        ax.set_title(f'{dataset_name} Dataset - {feature_label}', fontsize=12, fontweight='bold')
        ax.set_xlabel(feature_label, fontsize=10)
        ax.set_ylabel('Frequency', fontsize=10)
        ax.legend(fontsize=9, loc='upper right')
        ax.grid(True, alpha=0.3)

# 调整布局
plt.tight_layout()

# 保存图片
output_path = 'selected_features_distribution_comparison.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"可视化图片已保存为: {output_path}")

# 关闭图形以释放内存
plt.close()

print("完成！")

