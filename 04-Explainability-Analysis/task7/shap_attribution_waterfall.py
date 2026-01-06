#!/usr/bin/env python3
"""
SHAP Attribution Waterfall Plot for Domain Adaptation Models
基于SHAP的特征归因瀑布图可视化
分析35个手工提取特征对模型预测的贡献
"""
import os
import sys
import argparse
import numpy as np
import pandas as pd
import pickle
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, LabelEncoder

try:
    import shap
except ImportError:
    print("警告: SHAP库未安装。请运行: pip install shap")
    sys.exit(1)

# 添加models路径
sys.path.append(str(Path(__file__).parent / '域迁移模型' / 'models'))
from domain_adversarial_fusion_v3 import DomainAdversarialFusionMMD
from domain_adversarial_fusion_v7 import DomainAdversarialFusionV7

# 设置字体和绘图风格（期刊质量）
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

# 35个特征名称（基于特征提取详细说明.txt）
FEATURE_NAMES = [
    # 时域特征 (26个)
    'mean', 'maximum', 'minimum', 'variance', 'std',
    'skewness', 'kurtosis', 'median', 'q25', 'q75',
    'linear_slope', 'linear_intercept', 'linear_rss',
    'c3_0', 'c3_1', 'c3_2', 'c3_3',
    'c4_0', 'c4_1', 'c4_2', 'c4_3', 'c4_4',
    'absolute_sum_of_changes', 'mean_abs_change',
    'number_peaks', 'index_mass_quantile',
    # 频域特征 (4个)
    'spectral_centroid', 'spectral_bandwidth', 'spectral_rolloff', 'spectral_entropy',
    # 时频域特征 (5个)
    'wavelet_energy_A4', 'wavelet_energy_D4', 'wavelet_energy_D3', 
    'wavelet_energy_D2', 'wavelet_energy_D1'
]

LABEL_COLS = ['CrimpID', 'Wire_cross-section_conductor', 'Main_label_string',
              'Sub_label_string', 'Main-label_encoded', 'Sub_label_encoded',
              'Binary_label_encoded', 'CFM_label_encoded']


def load_datasets(base_dir: Path, source_size: str, target_size: str):
    """加载数据集"""
    datasets_dir = base_dir / '域迁移模型' / 'datasets'
    
    # 加载data1（原始曲线）
    source_data1 = pd.read_pickle(datasets_dir / 'data1' / f'crimp_force_curves_dataset_{source_size}.pkl')
    target_data1 = pd.read_pickle(datasets_dir / 'data1' / f'crimp_force_curves_dataset_{target_size}.pkl')
    
    # 加载data2（35个特征）
    source_data2 = pd.read_pickle(datasets_dir / 'data2' / f'features_{source_size}.pkl')
    target_data2 = pd.read_pickle(datasets_dir / 'data2' / f'features_{target_size}.pkl')
    
    return source_data1, source_data2, target_data1, target_data2


def prepare_data(data1, data2, label_col='Sub_label_encoded'):
    """准备数据：对齐并提取特征（与train_v3.py保持一致）"""
    # 使用CrimpID进行合并
    merged = pd.merge(data1, data2, on='CrimpID', how='inner', suffixes=('_data1', '_data2'))
    
    # 提取原始曲线数据
    raw_data = np.stack(merged['Force_curve_RoI'].values)
    
    # 排除列（与train_v3.py保持一致）
    exclude_cols = [
        'CrimpID', 'Wire_cross-section_conductor_data1', 'Wire_cross-section_conductor_data2',
        'Force_curve_raw', 'Force_curve_baseline', 'Force_curve_RoI',
        'Main_label_string_data1', 'Main_label_string_data2', 
        'Sub_label_string_data1', 'Sub_label_string_data2',
        'Main-label_encoded_data1', 'Main-label_encoded_data2',
        'Sub_label_encoded_data1', 'Sub_label_encoded_data2', 
        'Binary_label_encoded_data1', 'Binary_label_encoded_data2',
        'CFM_label_encoded_data1', 'CFM_label_encoded_data2'
    ]
    
    # 提取特征列
    feat_cols = [col for col in merged.columns if col not in exclude_cols]
    feat_data = merged[feat_cols].values.astype(np.float32)
    
    # 提取标签（使用Sub_label_encoded_data1，与train_v3.py保持一致）
    labels = merged['Sub_label_encoded_data1'].values
    
    return raw_data, feat_data, labels, feat_cols


class ModelWrapper:
    """模型包装类，用于SHAP分析"""
    
    def __init__(self, model, raw_data_sample, raw_scaler, feat_scaler, device):
        self.model = model
        self.raw_data_sample = raw_data_sample  # 固定的原始曲线数据（单个样本）
        self.raw_scaler = raw_scaler
        self.feat_scaler = feat_scaler
        self.device = device
        self.model.eval()
    
    def __call__(self, feat_vec):
        """
        SHAP解释器调用此函数
        Args:
            feat_vec: 特征向量 (n_samples, n_features) 或 (n_features,)
        Returns:
            预测概率 (n_samples, num_classes)
        """
        # 处理输入维度
        if feat_vec.ndim == 1:
            feat_vec = feat_vec.reshape(1, -1)
        
        batch_size = feat_vec.shape[0]
        
        # 标准化特征
        if self.feat_scaler:
            feat_vec_scaled = self.feat_scaler.transform(feat_vec)
        else:
            feat_vec_scaled = feat_vec
        
        # 转换为tensor
        feat_tensor = torch.tensor(feat_vec_scaled, dtype=torch.float32).to(self.device)
        
        # 准备原始曲线数据（固定样本，扩展到batch_size）
        raw_sample = self.raw_data_sample.copy()
        
        # 标准化原始曲线（单个样本）
        if self.raw_scaler:
            raw_sample_scaled = self.raw_scaler.transform(raw_sample.reshape(1, -1)).reshape(-1)
        else:
            raw_sample_scaled = raw_sample
        
        # 扩展到batch_size
        raw_tensor = torch.tensor(raw_sample_scaled, dtype=torch.float32).to(self.device)
        raw_tensor = raw_tensor.unsqueeze(0).repeat(batch_size, 1)
        
        # 前向传播
        with torch.no_grad():
            # 清理GPU缓存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            outputs = self.model(raw_tensor, feat_tensor, return_features=False)
            if isinstance(outputs, tuple):
                label_logits = outputs[0]
            else:
                label_logits = outputs
            
            # 返回softmax概率
            probs = F.softmax(label_logits, dim=1)
            
            # 立即清理
            del raw_tensor, feat_tensor, outputs, label_logits
        
        result = probs.cpu().numpy()
        del probs
        return result


def load_model(model_version: str, source_size: str, target_size: str, base_dir: Path, device):
    """加载训练好的模型"""
    experiments_dir = base_dir / '域迁移模型' / 'experiments'
    
    # 加载数据以获取维度信息（不需要data1，只需要data2获取特征维度）
    _, source_data2, _, target_data2 = load_datasets(base_dir, source_size, target_size)
    
    # 直接从data2获取特征列（排除标签列）
    exclude_cols = [
        'CrimpID', 'Wire_cross-section_conductor',
        'Main_label_string', 'Sub_label_string',
        'Main-label_encoded', 'Sub_label_encoded', 
        'Binary_label_encoded', 'CFM_label_encoded',
        'Force_curve_raw', 'Force_curve_baseline', 'Force_curve_RoI'
    ]
    feat_cols = [col for col in source_data2.columns if col not in exclude_cols]
    feat_dim = len(feat_cols)
    
    # 获取类别数：使用源域和目标域的并集（因为模型可能见过所有类别）
    source_labels = set(source_data2['Sub_label_encoded'].unique())
    target_labels = set(target_data2['Sub_label_encoded'].unique())
    all_labels = source_labels | target_labels
    num_classes = len(all_labels)
    
    if model_version == 'v3':
        model_dir = experiments_dir / 'v3' / f'{source_size}_to_{target_size}'
        model_path = model_dir / 'best_model.pt'
        
        model = DomainAdversarialFusionMMD(
            feat_in_dim=feat_dim,
            num_classes=num_classes,
            num_domains=2
        ).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=False))
        
    elif model_version == 'v7':
        model_dir = experiments_dir / 'v7' / f'{source_size}_to_{target_size}'
        model_path = model_dir / 'best_model.pt'
        ema_path = model_dir / 'best_model_ema.pt'
        
        model = DomainAdversarialFusionV7(
            feat_in_dim=feat_dim,
            num_classes=num_classes,
            num_domains=2
        ).to(device)
        
        # 优先使用EMA权重
        if ema_path.exists():
            print(f"使用EMA权重: {ema_path}")
            model.load_state_dict(torch.load(ema_path, map_location=device, weights_only=False))
        else:
            print(f"使用原始权重: {model_path}")
            model.load_state_dict(torch.load(model_path, map_location=device, weights_only=False))
    else:
        raise ValueError(f"未知的模型版本: {model_version}")
    
    print(f"已加载模型: {model_path}")
    print(f"特征维度: {feat_dim}, 类别数: {num_classes}")
    return model


def generate_shap_summary_plot(model, background_feat, sample_feat, sample_raw_data, 
                              raw_scaler, feat_scaler, feature_names, num_classes, 
                              device, output_file=None):
    """
    生成SHAP归因汇总图（所有类别，所有特征）
    使用多个样本计算SHAP值，显示整体特征重要性
    
    Args:
        model: PyTorch模型
        background_feat: 背景特征数据集（用于解释）
        sample_feat: 要解释的样本特征数据
        sample_raw_data: 要解释的样本原始曲线数据（每个样本对应一个）
        raw_scaler: 原始曲线标准化器
        feat_scaler: 特征标准化器
        feature_names: 特征名称列表
        num_classes: 类别数量
        device: 设备
        output_file: 输出文件路径
    """
    print(f"\n生成SHAP归因汇总图（所有类别，{len(feature_names)}个特征）...")
    print(f"背景特征数据集形状: {background_feat.shape}")
    print(f"样本特征数据形状: {sample_feat.shape}")
    print(f"样本原始曲线数据形状: {sample_raw_data.shape}")
    print(f"类别数: {num_classes}")
    
    # 定义解释函数：使用对应的原始曲线，只改变特征
    def explain_fn_max_prob(feat_vec):
        # feat_vec是特征向量 (n_samples, n_features)
        # 我们需要为每个特征向量使用对应的原始曲线
        batch_size = feat_vec.shape[0] if feat_vec.ndim > 1 else 1
        if feat_vec.ndim == 1:
            feat_vec = feat_vec.reshape(1, -1)
            batch_size = 1
        
        # 标准化特征
        if feat_scaler:
            feat_vec_scaled = feat_scaler.transform(feat_vec)
        else:
            feat_vec_scaled = feat_vec
        
        feat_tensor = torch.tensor(feat_vec_scaled, dtype=torch.float32).to(device)
        
        # 为每个特征向量使用对应的原始曲线（使用索引来匹配）
        # 简化：使用第一个样本的原始曲线作为固定输入（SHAP会处理特征的变化）
        # 或者：为每个样本使用其对应的原始曲线
        if batch_size == 1:
            # 单个样本
            raw_sample = sample_raw_data[0]
        else:
            # 多个样本：使用前batch_size个样本的原始曲线
            raw_sample = sample_raw_data[min(batch_size-1, len(sample_raw_data)-1)]
        
        # 标准化原始曲线
        if raw_scaler:
            raw_scaled = raw_scaler.transform(raw_sample.reshape(1, -1)).reshape(-1)
        else:
            raw_scaled = raw_sample
        
        # 扩展到batch_size
        raw_tensor = torch.tensor(raw_scaled, dtype=torch.float32).to(device)
        raw_tensor = raw_tensor.unsqueeze(0).repeat(batch_size, 1)
        
        # 前向传播（确保模型在eval模式）
        model.eval()
        with torch.no_grad():
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            outputs = model(raw_tensor, feat_tensor, return_features=False)
            if isinstance(outputs, tuple):
                label_logits = outputs[0]
            else:
                label_logits = outputs
            probs = F.softmax(label_logits, dim=1)
        
        result = np.max(probs.cpu().numpy(), axis=1)  # 返回最大类别概率
        del raw_tensor, feat_tensor, outputs, label_logits, probs
        return result
    
    # 使用KernelExplainer（减少背景样本以节省内存）
    print("计算SHAP值（这可能需要一些时间）...")
    background_sample = background_feat[:min(30, len(background_feat))]  # 进一步减少背景样本
    
    # 先测试函数是否正常工作
    test_feat = background_sample[:2]
    test_result = explain_fn_max_prob(test_feat)
    print(f"解释函数测试输出形状: {test_result.shape}, 范围: [{test_result.min():.6f}, {test_result.max():.6f}]")
    
    explainer = shap.KernelExplainer(explain_fn_max_prob, background_sample)
    
    # 计算多个样本的SHAP值（减少样本数量以节省内存）
    n_samples_to_explain = min(10, len(sample_feat))  # 解释前10个样本
    explain_samples = sample_feat[:n_samples_to_explain]
    
    print(f"正在计算 {n_samples_to_explain} 个样本的SHAP值...")
    shap_values = explainer.shap_values(explain_samples, nsamples=200)  # 增加nsamples提高精度
    
    # shap_values是一个数组或列表
    print(f"SHAP值类型: {type(shap_values)}")
    print(f"SHAP值形状: {shap_values.shape if hasattr(shap_values, 'shape') else 'N/A'}")
    
    # 打印原始SHAP值统计
    if isinstance(shap_values, np.ndarray):
        print(f"原始SHAP值统计: min={shap_values.min():.6f}, max={shap_values.max():.6f}, mean={shap_values.mean():.6f}, std={shap_values.std():.6f}")
        print(f"SHAP值非零数量: {np.count_nonzero(shap_values)} / {shap_values.size}")
    elif isinstance(shap_values, list):
        for i, sv in enumerate(shap_values):
            print(f"类别{i} SHAP值统计: min={sv.min():.6f}, max={sv.max():.6f}, mean={sv.mean():.6f}, std={sv.std():.6f}")
    
    if isinstance(shap_values, list):
        print(f"SHAP值列表长度: {len(shap_values)}")
        print(f"每个元素的SHAP值形状: {shap_values[0].shape}")
        # 计算平均绝对SHAP值作为整体重要性
        shap_values_mean = np.mean([np.abs(sv) for sv in shap_values], axis=0)  # (n_samples, n_features)
        shap_values_combined = shap_values_mean
    else:
        # 如果是数组，检查维度
        print(f"SHAP值数组形状: {shap_values.shape}")
        # 现在shap_values应该是 (n_samples, n_features)
        if shap_values.ndim == 2:
            shap_values_combined = np.abs(shap_values)  # (n_samples, n_features)
        else:
            shap_values_combined = np.abs(shap_values.reshape(-1, len(feature_names)))
    
    print(f"合并后SHAP值形状: {shap_values_combined.shape}")
    
    # 计算每个特征的平均重要性（跨所有样本）
    if shap_values_combined.ndim == 2:
        feature_importance = np.mean(np.abs(shap_values_combined), axis=0)  # (n_features,)
    else:
        feature_importance = np.abs(shap_values_combined)
    
    print(f"特征重要性形状: {feature_importance.shape}")
    
    print(f"特征重要性形状: {feature_importance.shape}")
    print(f"特征重要性范围: [{feature_importance.min():.6f}, {feature_importance.max():.6f}]")
    
    # 创建美观的柱状图
    fig, ax = plt.subplots(figsize=(14, 10))
    fig.patch.set_facecolor('white')
    
    # 按重要性排序
    sorted_indices = np.argsort(feature_importance)[::-1]
    sorted_importance = feature_importance[sorted_indices]
    sorted_names = [feature_names[int(i)] for i in sorted_indices]  # 确保转换为整数索引
    
    # 创建颜色映射（从蓝色到红色）
    colors = plt.cm.RdYlBu_r(np.linspace(0.2, 0.8, len(sorted_importance)))
    
    # 绘制水平柱状图
    bars = ax.barh(range(len(sorted_names)), sorted_importance, color=colors, edgecolor='white', linewidth=1.5)
    
    # 设置标签
    ax.set_yticks(range(len(sorted_names)))
    ax.set_yticklabels(sorted_names, fontsize=10)
    ax.set_xlabel('Mean |SHAP Value| (Feature Importance)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Features', fontsize=12, fontweight='bold')
    ax.set_title(f'SHAP Feature Attribution Summary\n(All Classes, {len(feature_names)} Features)', 
                 fontsize=14, fontweight='bold', pad=20)
    
    # 添加网格
    ax.grid(True, axis='x', linestyle='--', alpha=0.3, color='gray')
    ax.set_axisbelow(True)
    
    # 反转y轴，使最重要的特征在顶部
    ax.invert_yaxis()
    
    # 美化
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)
    
    # 在每个柱子上添加数值标签
    for i, (bar, val) in enumerate(zip(bars, sorted_importance)):
        ax.text(val + max(sorted_importance) * 0.01, bar.get_y() + bar.get_height()/2, 
               f'{val:.4f}', va='center', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"SHAP汇总图已保存: {output_file}")
    
    plt.close()
    
    # 也创建一个summary plot（SHAP内置）
    try:
        fig2, ax2 = plt.subplots(figsize=(12, 10))
        
        # 使用第一个类别的SHAP值创建summary plot（作为示例）
        if isinstance(shap_values, list):
            shap_summary = shap_values[0]  # 使用第一个类别
        else:
            shap_summary = shap_values
        
        # 创建Explanation对象
        shap_explanation = shap.Explanation(
            values=shap_summary,
            base_values=explainer.expected_value[0] if isinstance(explainer.expected_value, np.ndarray) else explainer.expected_value,
            data=explain_samples,
            feature_names=feature_names
        )
        
        shap.plots.beeswarm(shap_explanation, max_display=35, show=False, ax=ax2)
        ax2.set_title(f'SHAP Summary Plot (All {len(feature_names)} Features)', 
                     fontsize=14, fontweight='bold', pad=20)
        plt.tight_layout()
        
        summary_file = output_file.replace('.png', '_summary.png') if output_file else None
        if summary_file:
            plt.savefig(summary_file, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"SHAP Summary Plot已保存: {summary_file}")
        plt.close()
    except Exception as e:
        print(f"创建Summary Plot时出错: {e}")
    
    return shap_values_combined, feature_importance


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='SHAP Attribution Waterfall Plot for Domain Adaptation')
    parser.add_argument('--direction', type=str, choices=['05_to_035', '035_to_05', 'both'],
                       default='both', help='迁移方向')
    parser.add_argument('--output_dir', type=str, default='shap_waterfall',
                       help='输出目录')
    parser.add_argument('--n_samples', type=int, default=50,
                       help='用于SHAP分析的样本数量')
    
    args = parser.parse_args()
    
    base_dir = Path(__file__).parent
    output_dir = base_dir / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    directions_to_process = []
    if args.direction in ['05_to_035', 'both']:
        directions_to_process.append(('05_to_035', '05', '035', 'v3'))
    if args.direction in ['035_to_05', 'both']:
        directions_to_process.append(('035_to_05', '035', '05', 'v7'))
    
    try:
        for direction_key, source_size, target_size, model_version in directions_to_process:
            print(f"\n{'='*60}")
            print(f"处理迁移方向: {source_size} → {target_size}")
            print(f"模型版本: {model_version}")
            print(f"{'='*60}")
            
            # 加载数据
            source_data1, source_data2, target_data1, target_data2 = load_datasets(
                base_dir, source_size, target_size
            )
            
            # 准备源域数据
            source_raw, source_feat, source_labels, feat_cols = prepare_data(
                source_data1, source_data2
            )
            
            # 准备目标域数据
            target_raw, target_feat, target_labels, _ = prepare_data(
                target_data1, target_data2
            )
            
            print(f"源域特征形状: {source_feat.shape}")
            print(f"目标域特征形状: {target_feat.shape}")
            print(f"特征数量: {len(feat_cols)}")
            
            # 验证特征名称
            if len(feat_cols) != 35:
                print(f"警告: 特征数量为{len(feat_cols)}，不是35个")
                feature_names = feat_cols[:35] if len(feat_cols) >= 35 else feat_cols + [''] * (35 - len(feat_cols))
            else:
                feature_names = feat_cols
            
            # 数据标准化
            raw_scaler = StandardScaler()
            raw_scaler.fit(np.concatenate([source_raw, target_raw], axis=0))
            
            feat_scaler = StandardScaler()
            feat_scaler.fit(np.concatenate([source_feat, target_feat], axis=0))
            
            # 加载模型
            model = load_model(model_version, source_size, target_size, base_dir, device)
            
            # 准备背景数据和样本数据（减少数量以节省内存）
            n_samples = min(20, len(target_feat))  # 限制最多20个样本
            background_feat = feat_scaler.transform(target_feat[:100])  # 使用100个目标域样本作为背景
            sample_feat = feat_scaler.transform(target_feat[:n_samples])  # 使用多个样本进行分析
            sample_raw_data = target_raw[:n_samples]  # 对应的原始曲线数据
            
            # 获取类别数
            num_classes = len(np.unique(target_labels))
            
            # 生成SHAP汇总图（所有类别，所有特征）
            output_file = output_dir / f'SHAP_Summary_{source_size}_to_{target_size}_AllClasses_AllFeatures.png'
            shap_values, feature_importance = generate_shap_summary_plot(
                model, background_feat, sample_feat, sample_raw_data,
                raw_scaler, feat_scaler, feature_names,
                num_classes=num_classes, device=device, output_file=output_file
            )
            
            # 打印特征重要性排序
            print(f"\n特征重要性排序 (Top 10):")
            sorted_indices = np.argsort(feature_importance)[::-1]
            for i, idx in enumerate(sorted_indices[:10]):
                print(f"  {i+1}. {feature_names[idx]}: {feature_importance[idx]:.6f}")
        
        print(f"\n{'='*60}")
        print("SHAP归因分析完成!")
        print(f"结果保存在: {output_dir}")
        print(f"{'='*60}")
        
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

