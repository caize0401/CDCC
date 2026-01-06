#!/usr/bin/env python3
"""
特征提取脚本
从域迁移模型和基础MLP模型提取融合特征并保存
"""
import os
import sys
import argparse
import numpy as np
import pandas as pd
import pickle
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, LabelEncoder

# 添加路径
sys.path.append(str(Path(__file__).parent / '域迁移模型' / 'models'))
sys.path.append(str(Path(__file__).parent / '基础模型'))

from domain_adversarial_fusion_v3 import DomainAdversarialFusionMMD
from domain_adversarial_fusion_v7 import DomainAdversarialFusionV7
from mlp_pytorch import BaseMLP, SimpleDataset, align_and_prepare


def load_datasets(base_dir: Path, source_size: str, target_size: str):
    """加载数据集"""
    datasets_dir = base_dir / '域迁移模型' / 'datasets'
    
    # 加载源域数据
    source_data1_path = datasets_dir / 'data1' / f'crimp_force_curves_dataset_{source_size}.pkl'
    source_data2_path = datasets_dir / 'data2' / f'features_{source_size}.pkl'
    
    # 加载目标域数据
    target_data1_path = datasets_dir / 'data1' / f'crimp_force_curves_dataset_{target_size}.pkl'
    target_data2_path = datasets_dir / 'data2' / f'features_{target_size}.pkl'
    
    with open(source_data1_path, 'rb') as f:
        source_data1 = pickle.load(f)
    with open(source_data2_path, 'rb') as f:
        source_data2 = pickle.load(f)
    with open(target_data1_path, 'rb') as f:
        target_data1 = pickle.load(f)
    with open(target_data2_path, 'rb') as f:
        target_data2 = pickle.load(f)
    
    return source_data1, source_data2, target_data1, target_data2


def extract_features_from_model(model, data_loader, device, model_type='base'):
    """从模型提取融合特征"""
    model.eval()
    all_features = []
    all_labels = []
    all_domains = []
    
    with torch.no_grad():
        for batch in data_loader:
            if model_type == 'base':
                raw, feat, label = batch
                domain = None
            else:
                raw, feat, label, domain = batch
            
            raw = raw.to(device)
            feat = feat.to(device)
            
            # 前向传播获取特征
            if model_type == 'base':
                _, features = model(raw, feat, return_features=True)
            else:
                _, _, _, features = model(raw, feat, return_features=True)
            
            all_features.append(features.cpu().numpy())
            all_labels.append(label.numpy())
            if domain is not None:
                all_domains.append(domain.numpy())
    
    features = np.concatenate(all_features, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    if all_domains:
        domains = np.concatenate(all_domains, axis=0)
    else:
        domains = None
    
    return features, labels, domains


def extract_base_mlp_features(source_size: str, target_size: str, base_dir: Path, output_dir: Path):
    """从基础MLP模型提取特征（迁移前特征）"""
    print(f"\n{'='*60}")
    print(f"提取基础MLP特征: {source_size} → {target_size} (迁移前)")
    print(f"{'='*60}")
    
    # 加载数据
    source_data1, source_data2, target_data1, target_data2 = load_datasets(
        base_dir, source_size, target_size
    )
    
    # 对齐和准备数据
    X_raw_source, X_feat_source, y_source = align_and_prepare(source_data1, source_data2)
    X_raw_target, X_feat_target, y_target = align_and_prepare(target_data1, target_data2)
    
    # 标签编码
    label_encoder = LabelEncoder()
    all_labels = np.concatenate([y_source, y_target])
    label_encoder.fit(all_labels)
    
    y_source_encoded = label_encoder.transform(y_source)
    y_target_encoded = label_encoder.transform(y_target)
    
    num_classes = len(label_encoder.classes_)
    
    # 合并所有数据用于提取特征
    X_raw_all = np.concatenate([X_raw_source, X_raw_target], axis=0)
    X_feat_all = np.concatenate([X_feat_source, X_feat_target], axis=0)
    y_all = np.concatenate([y_source_encoded, y_target_encoded], axis=0)
    domain_all = np.concatenate([
        np.zeros(len(y_source_encoded), dtype=int),  # 源域 = 0
        np.ones(len(y_target_encoded), dtype=int)    # 目标域 = 1
    ], axis=0)
    
    # 数据归一化（使用源域数据拟合）
    raw_scaler = StandardScaler().fit(X_raw_source)
    feat_scaler = StandardScaler().fit(X_feat_source)
    
    # 创建数据集和数据加载器
    dataset = SimpleDataset(X_raw_all, X_feat_all, y_all, raw_scaler, feat_scaler)
    data_loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=0)
    
    # 加载模型
    model_dir = base_dir / '基础模型' / 'experiments' / f'{source_size}_to_{target_size}'
    model_path = model_dir / 'best_model.pt'
    
    if not model_path.exists():
        raise FileNotFoundError(f"模型文件不存在: {model_path}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = BaseMLP(
        feat_in_dim=X_feat_source.shape[1],
        num_classes=num_classes,
        raw_out_dim=256,
        feat_out_dim=128,
        fusion_hidden=384
    ).to(device)
    
    model.load_state_dict(torch.load(model_path, weights_only=False))
    print(f"已加载模型: {model_path}")
    
    # 提取特征
    features, labels, _ = extract_features_from_model(model, data_loader, device, model_type='base')
    
    # 分离源域和目标域特征
    source_mask = domain_all == 0
    target_mask = domain_all == 1
    
    source_features = features[source_mask]
    target_features = features[target_mask]
    source_labels = labels[source_mask]
    target_labels = labels[target_mask]
    
    print(f"源域特征形状: {source_features.shape}")
    print(f"目标域特征形状: {target_features.shape}")
    
    # 保存特征
    output_file = output_dir / f'base_mlp_features_{source_size}_to_{target_size}.pkl'
    with open(output_file, 'wb') as f:
        pickle.dump({
            'source_features': source_features,
            'target_features': target_features,
            'source_labels': source_labels,
            'target_labels': target_labels,
            'source_size': source_size,
            'target_size': target_size,
            'model_type': 'base_mlp'
        }, f)
    
    print(f"特征已保存到: {output_file}")
    
    return source_features, target_features, source_labels, target_labels


def extract_domain_adaptation_features(model_version: str, source_size: str, target_size: str, 
                                      base_dir: Path, output_dir: Path):
    """从域迁移模型提取特征（迁移后特征）"""
    print(f"\n{'='*60}")
    print(f"提取域迁移模型特征: {model_version} - {source_size} → {target_size} (迁移后)")
    print(f"{'='*60}")
    
    # 加载数据
    source_data1, source_data2, target_data1, target_data2 = load_datasets(
        base_dir, source_size, target_size
    )
    
    # 对齐和准备数据
    X_raw_source, X_feat_source, y_source = align_and_prepare(source_data1, source_data2)
    X_raw_target, X_feat_target, y_target = align_and_prepare(target_data1, target_data2)
    
    # 标签编码
    label_encoder = LabelEncoder()
    all_labels = np.concatenate([y_source, y_target])
    label_encoder.fit(all_labels)
    
    y_source_encoded = label_encoder.transform(y_source)
    y_target_encoded = label_encoder.transform(y_target)
    
    num_classes = len(label_encoder.classes_)
    
    # 合并所有数据用于提取特征
    X_raw_all = np.concatenate([X_raw_source, X_raw_target], axis=0)
    X_feat_all = np.concatenate([X_feat_source, X_feat_target], axis=0)
    y_all = np.concatenate([y_source_encoded, y_target_encoded], axis=0)
    domain_all = np.concatenate([
        np.zeros(len(y_source_encoded), dtype=int),  # 源域 = 0
        np.ones(len(y_target_encoded), dtype=int)    # 目标域 = 1
    ], axis=0)
    
    # 数据归一化（使用源域和目标域合并数据拟合，与训练时一致）
    X_raw_train = np.concatenate([X_raw_source, X_raw_target], axis=0)
    X_feat_train = np.concatenate([X_feat_source, X_feat_target], axis=0)
    raw_scaler = StandardScaler().fit(X_raw_train)
    feat_scaler = StandardScaler().fit(X_feat_train)
    
    # 创建数据集（使用域迁移模型的Dataset类）
    class DomainAdversarialDatasetMMD(Dataset):
        """MMD版本的域对抗数据集"""
        
        def __init__(self, raw_data, feat_data, labels, domains, raw_scaler=None, feat_scaler=None):
            self.raw_data = raw_data
            self.feat_data = feat_data
            self.labels = labels
            self.domains = domains
            self.raw_scaler = raw_scaler
            self.feat_scaler = feat_scaler
        
        def __len__(self):
            return len(self.raw_data)
        
        def __getitem__(self, idx):
            raw = self.raw_data[idx].astype(np.float32)
            feat = self.feat_data[idx].astype(np.float32)
            label = self.labels[idx]
            domain = self.domains[idx]
            
            # 归一化
            if self.raw_scaler:
                raw = self.raw_scaler.transform(raw.reshape(1, -1)).reshape(-1)
            if self.feat_scaler:
                feat = self.feat_scaler.transform(feat.reshape(1, -1)).reshape(-1)
            
            return (
                torch.tensor(raw, dtype=torch.float32),
                torch.tensor(feat, dtype=torch.float32),
                torch.tensor(label, dtype=torch.long),
                torch.tensor(domain, dtype=torch.long)
            )
    
    dataset = DomainAdversarialDatasetMMD(
        X_raw_all, X_feat_all, y_all, domain_all, raw_scaler, feat_scaler
    )
    data_loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=0)
    
    # 加载模型
    model_dir = base_dir / '域迁移模型' / 'experiments' / model_version / f'{source_size}_to_{target_size}'
    model_path = model_dir / 'best_model.pt'
    
    if not model_path.exists():
        raise FileNotFoundError(f"模型文件不存在: {model_path}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if model_version == 'v3':
        model = DomainAdversarialFusionMMD(
            feat_in_dim=X_feat_source.shape[1],
            num_classes=num_classes,
            num_domains=2
        ).to(device)
        model.load_state_dict(torch.load(model_path, weights_only=False))
    elif model_version == 'v7':
        model = DomainAdversarialFusionV7(
            feat_in_dim=X_feat_source.shape[1],
            num_classes=num_classes,
            num_domains=2
        ).to(device)
        # v7模型优先使用EMA权重
        ema_path = model_dir / 'best_model_ema.pt'
        if ema_path.exists():
            print(f"使用EMA权重: {ema_path}")
            model.load_state_dict(torch.load(ema_path, weights_only=False))
        else:
            print(f"使用原始权重: {model_path}")
            model.load_state_dict(torch.load(model_path, weights_only=False))
    else:
        raise ValueError(f"未知的模型版本: {model_version}")
    
    print(f"已加载模型: {model_path}")
    
    # 提取特征
    features, labels, domains = extract_features_from_model(
        model, data_loader, device, model_type='domain_adaptation'
    )
    
    # 分离源域和目标域特征
    source_mask = domains == 0
    target_mask = domains == 1
    
    source_features = features[source_mask]
    target_features = features[target_mask]
    source_labels = labels[source_mask]
    target_labels = labels[target_mask]
    
    print(f"源域特征形状: {source_features.shape}")
    print(f"目标域特征形状: {target_features.shape}")
    
    # 保存特征
    output_file = output_dir / f'{model_version}_features_{source_size}_to_{target_size}.pkl'
    with open(output_file, 'wb') as f:
        pickle.dump({
            'source_features': source_features,
            'target_features': target_features,
            'source_labels': source_labels,
            'target_labels': target_labels,
            'source_size': source_size,
            'target_size': target_size,
            'model_type': model_version
        }, f)
    
    print(f"特征已保存到: {output_file}")
    
    return source_features, target_features, source_labels, target_labels


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='特征提取脚本')
    parser.add_argument('--mode', type=str, required=True, choices=['base', 'v3', 'v7', 'all'],
                       help='提取模式: base=基础MLP, v3=域迁移v3, v7=域迁移v7, all=全部')
    parser.add_argument('--output', type=str, default='features',
                       help='特征保存目录')
    
    args = parser.parse_args()
    
    base_dir = Path(__file__).parent
    output_dir = Path(args.output)
    if not output_dir.is_absolute():
        output_dir = base_dir / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 添加域迁移模型路径
    sys.path.append(str(base_dir / '域迁移模型'))
    
    print("=" * 60)
    print("特征提取脚本")
    print("=" * 60)
    print(f"模式: {args.mode}")
    print(f"输出目录: {output_dir}")
    print("=" * 60)
    
    try:
        if args.mode == 'base' or args.mode == 'all':
            # 提取基础MLP特征（两个方向）
            print("\n提取基础MLP特征...")
            extract_base_mlp_features('05', '035', base_dir, output_dir)
            extract_base_mlp_features('035', '05', base_dir, output_dir)
        
        if args.mode == 'v3' or args.mode == 'all':
            # 提取v3特征
            print("\n提取v3特征...")
            extract_domain_adaptation_features('v3', '05', '035', base_dir, output_dir)
        
        if args.mode == 'v7' or args.mode == 'all':
            # 提取v7特征
            print("\n提取v7特征...")
            extract_domain_adaptation_features('v7', '035', '05', base_dir, output_dir)
        
        print(f"\n{'='*60}")
        print("特征提取完成！")
        print(f"特征保存在: {output_dir}")
        print(f"{'='*60}")
        
    except Exception as e:
        print(f"\n特征提取失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

