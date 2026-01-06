import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import random
import pickle
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def create_output_folder():
    """创建输出文件夹"""
    output_folder = "task1/curve_visualizations_real_data"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"创建输出文件夹: {output_folder}")
    return output_folder

def load_dataset(dataset_path):
    """加载数据集"""
    try:
        with open(dataset_path, 'rb') as f:
            df = pickle.load(f)
        print(f"成功加载数据集: {dataset_path}")
        print(f"数据集形状: {df.shape}")
        return df
    except Exception as e:
        print(f"加载数据集失败: {e}")
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

def visualize_curve(curve_data, title, output_path, dataset_type, sample_idx, label_info=""):
    """可视化单条曲线"""
    plt.figure(figsize=(14, 10))
    
    # 绘制曲线
    plt.plot(curve_data, linewidth=2, color='blue', alpha=0.8)
    
    # 设置标题和标签
    plt.title(f'{title}\n数据集: {dataset_type}, 样本: {sample_idx+1}', fontsize=16, fontweight='bold')
    plt.xlabel('数据点索引', fontsize=14)
    plt.ylabel('力值', fontsize=14)
    
    # 添加网格
    plt.grid(True, alpha=0.3)
    
    # 设置坐标轴
    plt.xlim(0, len(curve_data)-1)
    
    # 添加统计信息
    mean_val = np.mean(curve_data)
    max_val = np.max(curve_data)
    min_val = np.min(curve_data)
    std_val = np.std(curve_data)
    
    # 计算有效数据点（非-9或-30的值）
    valid_data = curve_data[~np.isin(curve_data, [-9, -30])]
    if len(valid_data) > 0:
        valid_mean = np.mean(valid_data)
        valid_max = np.max(valid_data)
        valid_min = np.min(valid_data)
        valid_std = np.std(valid_data)
        valid_count = len(valid_data)
    else:
        valid_mean = valid_max = valid_min = valid_std = 0
        valid_count = 0
    
    stats_text = f'全部数据统计:\n均值: {mean_val:.2f}\n最大值: {max_val:.2f}\n最小值: {min_val:.2f}\n标准差: {std_val:.2f}'
    if valid_count > 0:
        stats_text += f'\n\n有效数据统计:\n有效点数: {valid_count}\n均值: {valid_mean:.2f}\n最大值: {valid_max:.2f}\n最小值: {valid_min:.2f}\n标准差: {valid_std:.2f}'
    
    if label_info:
        stats_text += f'\n\n{label_info}'
    
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
             fontsize=11)
    
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
            
            # 获取标签信息
            label_info = ""
            if 'CrimpID' in row:
                label_info += f"CrimpID: {row['CrimpID']}"
            if 'Main_label_string' in row:
                label_info += f"\n主标签: {row['Main_label_string']}"
            if 'Sub_label_string' in row:
                label_info += f"\n子标签: {row['Sub_label_string']}"
            if 'Wire_cross-section_conductor' in row:
                label_info += f"\n导线截面: {row['Wire_cross-section_conductor']}"
            
            # 创建标题
            title = f'力曲线可视化 - {dataset_name}数据集 (真实数据)'
            
            # 创建输出文件名
            output_filename = f'{dataset_name}_sample_{idx+1:02d}_real.png'
            output_path = os.path.join(output_folder, output_filename)
            
            # 可视化曲线
            visualize_curve(curve_data, title, output_path, dataset_name, idx, label_info)
            
        except Exception as e:
            print(f"可视化样本{idx+1}时出错: {e}")
            continue

def main():
    """主函数"""
    print("开始真实数据曲线可视化任务...")
    
    # 设置随机种子以确保结果可重现
    random.seed(42)
    np.random.seed(42)
    
    # 创建输出文件夹
    output_folder = create_output_folder()
    
    # 数据集路径
    dataset_05_path = "datasets/crimp_force_curves_dataset_05.pkl"
    dataset_035_path = "datasets/crimp_force_curves_dataset_035.pkl"
    
    # 加载0.5数据集
    print("\n" + "="*60)
    print("加载0.5数据集")
    print("="*60)
    df_05 = load_dataset(dataset_05_path)
    if df_05 is not None:
        print(f"0.5数据集列名: {df_05.columns.tolist()}")
        visualize_dataset_curves(df_05, "0.5", output_folder, n_samples=5)
    else:
        print("无法加载0.5数据集，跳过...")
    
    # 加载0.35数据集
    print("\n" + "="*60)
    print("加载0.35数据集")
    print("="*60)
    df_035 = load_dataset(dataset_035_path)
    if df_035 is not None:
        print(f"0.35数据集列名: {df_035.columns.tolist()}")
        visualize_dataset_curves(df_035, "0.35", output_folder, n_samples=5)
    else:
        print("无法加载0.35数据集，跳过...")
    
    print("\n" + "="*60)
    print("真实数据可视化任务完成！")
    print(f"所有图片已保存到: {output_folder}")
    print("="*60)

if __name__ == "__main__":
    main()














