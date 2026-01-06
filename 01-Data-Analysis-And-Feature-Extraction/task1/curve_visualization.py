import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import random
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def create_output_folder():
    """创建输出文件夹"""
    output_folder = "task1/curve_visualizations"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"创建输出文件夹: {output_folder}")
    return output_folder

def load_dataset(dataset_path):
    """加载数据集"""
    try:
        # 尝试使用pickle直接加载
        import pickle
        with open(dataset_path, 'rb') as f:
            df = pickle.load(f)
        print(f"成功加载数据集: {dataset_path}")
        print(f"数据集形状: {df.shape}")
        return df
    except Exception as e:
        print(f"使用pickle加载失败: {e}")
        try:
            # 尝试使用pandas加载
            df = pd.read_pickle(dataset_path)
            print(f"使用pandas成功加载数据集: {dataset_path}")
            print(f"数据集形状: {df.shape}")
            return df
        except Exception as e2:
            print(f"使用pandas加载也失败: {e2}")
            return None

def random_sample_data(df, n_samples=5):
    """从数据集中随机采样n_samples条数据"""
    if len(df) < n_samples:
        print(f"警告: 数据集只有{len(df)}条数据，少于请求的{n_samples}条")
        n_samples = len(df)
    
    # 随机选择索引
    random_indices = random.sample(range(len(df)), n_samples)
    sampled_data = df.iloc[random_indices]
    
    print(f"随机选择了{len(sampled_data)}条数据")
    return sampled_data

def visualize_curve(curve_data, title, output_path, dataset_type, sample_idx):
    """可视化单条曲线"""
    plt.figure(figsize=(12, 8))
    
    # 绘制曲线
    plt.plot(curve_data, linewidth=2, color='blue', alpha=0.8)
    
    # 设置标题和标签
    plt.title(f'{title}\n数据集: {dataset_type}, 样本: {sample_idx+1}', fontsize=14, fontweight='bold')
    plt.xlabel('数据点索引', fontsize=12)
    plt.ylabel('力值', fontsize=12)
    
    # 添加网格
    plt.grid(True, alpha=0.3)
    
    # 设置坐标轴
    plt.xlim(0, len(curve_data)-1)
    
    # 添加统计信息
    mean_val = np.mean(curve_data)
    max_val = np.max(curve_data)
    min_val = np.min(curve_data)
    std_val = np.std(curve_data)
    
    stats_text = f'均值: {mean_val:.2f}\n最大值: {max_val:.2f}\n最小值: {min_val:.2f}\n标准差: {std_val:.2f}'
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
             fontsize=10)
    
    # 保存图片
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"图片已保存: {output_path}")

def visualize_dataset_curves(df, dataset_name, output_folder, n_samples=5):
    """可视化数据集中的曲线"""
    print(f"\n开始可视化{dataset_name}数据集...")
    
    # 随机采样数据
    sampled_data = random_sample_data(df, n_samples)
    
    # 可视化每条曲线
    for idx, (_, row) in enumerate(sampled_data.iterrows()):
        try:
            # 获取曲线数据
            curve_data = row['Force_curve_RoI']
            
            # 确保曲线数据是numpy数组
            if not isinstance(curve_data, np.ndarray):
                curve_data = np.array(curve_data)
            
            # 检查数据长度
            if len(curve_data) != 500:
                print(f"警告: 样本{idx+1}的曲线长度是{len(curve_data)}，不是500")
            
            # 创建标题
            title = f'力曲线可视化 - {dataset_name}数据集'
            
            # 创建输出文件名
            output_filename = f'{dataset_name}_sample_{idx+1:02d}.png'
            output_path = os.path.join(output_folder, output_filename)
            
            # 可视化曲线
            visualize_curve(curve_data, title, output_path, dataset_name, idx)
            
        except Exception as e:
            print(f"可视化样本{idx+1}时出错: {e}")
            continue

def main():
    """主函数"""
    print("开始曲线可视化任务...")
    
    # 设置随机种子以确保结果可重现
    random.seed(42)
    np.random.seed(42)
    
    # 创建输出文件夹
    output_folder = create_output_folder()
    
    # 数据集路径
    dataset_05_path = "datasets/crimp_force_curves_dataset_05.pkl"
    dataset_035_path = "datasets/crimp_force_curves_dataset_035.pkl"
    
    # 加载0.5数据集
    print("\n" + "="*50)
    print("加载0.5数据集")
    print("="*50)
    df_05 = load_dataset(dataset_05_path)
    if df_05 is not None:
        visualize_dataset_curves(df_05, "0.5", output_folder, n_samples=5)
    
    # 加载0.35数据集
    print("\n" + "="*50)
    print("加载0.35数据集")
    print("="*50)
    df_035 = load_dataset(dataset_035_path)
    if df_035 is not None:
        visualize_dataset_curves(df_035, "0.35", output_folder, n_samples=5)
    
    print("\n" + "="*50)
    print("可视化任务完成！")
    print(f"所有图片已保存到: {output_folder}")
    print("="*50)

if __name__ == "__main__":
    main()
