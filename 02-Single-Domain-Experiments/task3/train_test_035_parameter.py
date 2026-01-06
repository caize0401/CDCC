#!/usr/bin/env python3
"""
Hybrid XGBoost v11 Training Script
端到端训练的混合模型，使用XGBoost预测作为损失信号
"""
import os
from pathlib import Path
import argparse
import random
from typing import List, Tuple
import sys
import importlib
import inspect

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import xgboost as xgb

# 添加models路径
sys.path.append(str(Path(__file__).parent / 'models'))
from hybrid_Xgboost_v11 import HybridXGBoostV11

RANDOM_SEED: int = 42
torch.backends.cudnn.benchmark = True


LABEL_COLS: List[str] = [
    'CrimpID',
    'Wire_cross-section_conductor',
    'Main_label_string',
    'Sub_label_string',
    'Main-label_encoded',
    'Sub_label_encoded',
    'Binary_label_encoded',
    'CFM_label_encoded',
]


def set_seed(seed: int = RANDOM_SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_size_datasets(base_dir: Path, size_tag: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    data1_path = base_dir / 'datasets' / 'data1' / f'crimp_force_curves_dataset_{size_tag}.pkl'
    data2_path = base_dir / 'datasets' / 'data2' / f'features_{size_tag}.pkl'
    if not data1_path.exists():
        raise FileNotFoundError(f"Missing file: {data1_path}")
    if not data2_path.exists():
        raise FileNotFoundError(f"Missing file: {data2_path}")
    df_raw = pd.read_pickle(data1_path)
    df_feat = pd.read_pickle(data2_path)
    return df_raw, df_feat


def align_and_prepare_dual_path(df_raw: pd.DataFrame, df_feat: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """对齐并准备双路径数据"""
    # 对齐数据
    common_ids = set(df_raw['CrimpID']).intersection(set(df_feat['CrimpID']))
    df_raw_aligned = df_raw[df_raw['CrimpID'].isin(common_ids)].sort_values('CrimpID')
    df_feat_aligned = df_feat[df_feat['CrimpID'].isin(common_ids)].sort_values('CrimpID')
    
    # 提取原始曲线数据
    raw_curves = np.array([curve for curve in df_raw_aligned['Force_curve_RoI'].values])
    
    # 提取特征数据
    feat_cols = [col for col in df_feat_aligned.columns if col not in LABEL_COLS]
    feat_data = df_feat_aligned[feat_cols].values
    
    # 提取标签
    labels = df_feat_aligned['Sub_label_encoded'].values
    
    return raw_curves, feat_data, labels


def preprocess_dual_path_data(raw_curves: np.ndarray, feat_data: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, StandardScaler, StandardScaler]:
    """预处理双路径数据"""
    # 数据标准化
    raw_scaler = StandardScaler()
    feat_scaler = StandardScaler()
    
    raw_curves_scaled = raw_scaler.fit_transform(raw_curves)
    feat_data_scaled = feat_scaler.fit_transform(feat_data)
    
    return raw_curves_scaled, feat_data_scaled, labels, raw_scaler, feat_scaler


class DualPathDataset(Dataset):
    """双路径数据集"""
    def __init__(self, raw_data, feat_data, labels):
        self.raw_data = raw_data
        self.feat_data = feat_data
        self.labels = labels
    
    def __len__(self):
        return len(self.raw_data)
    
    def __getitem__(self, idx):
        return self.raw_data[idx], self.feat_data[idx], self.labels[idx]


def train_epoch(model, train_loader, optimizer, device, xgb_weight=0.5):
    """训练一个epoch"""
    model.train()
    total_loss = 0.0
    total_acc = 0.0
    total_n = 0
    
    for raw, feat, labels in train_loader:
        raw = raw.to(device)
        feat = feat.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        # 前向传播
        logits, fused_features = model(raw, feat)
        
        # 计算分类损失
        classification_loss = F.cross_entropy(logits, labels)
        
        # 计算XGBoost损失
        xgb_loss = model.compute_xgb_loss(raw, feat, labels)
        
        # 总损失：分类损失 + XGBoost损失
        total_loss_batch = classification_loss + xgb_weight * xgb_loss
        
        # 反向传播
        total_loss_batch.backward()
        optimizer.step()
        
        # 统计
        total_loss += total_loss_batch.item()
        pred = logits.argmax(dim=1)
        total_acc += (pred == labels).sum().item()
        total_n += labels.size(0)
    
    return total_loss / len(train_loader), total_acc / total_n


def evaluate_model(model, test_loader, device):
    """评估模型"""
    model.eval()
    all_preds = []
    all_labels = []
    total_loss = 0.0
    
    with torch.no_grad():
        for raw, feat, labels in test_loader:
            raw = raw.to(device)
            feat = feat.to(device)
            labels = labels.to(device)
            
            logits, _ = model(raw, feat)
            preds = logits.argmax(dim=1)
            
            # 计算损失
            loss = F.cross_entropy(logits, labels)
            total_loss += loss.item()
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # 计算指标
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    
    # 混淆矩阵
    cm = confusion_matrix(all_labels, all_preds)
    
    return {
        'loss': total_loss / len(test_loader),
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm
    }


def run(size_tag: str, out_dir: Path, epochs: int = 50, batch_size: int = 128, lr: float = 1e-3, xgb_weight: float = 0.5):
    """运行训练"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 加载数据
    print(f"加载 {size_tag} 数据集...")
    df_raw, df_feat = load_size_datasets(Path(__file__).parent, size_tag)
    
    # 对齐和准备数据
    raw_curves, feat_data, labels = align_and_prepare_dual_path(df_raw, df_feat)
    
    # 重新编码标签为连续整数
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    labels = le.fit_transform(labels)
    
    # 预处理数据
    raw_curves_scaled, feat_data_scaled, labels, raw_scaler, feat_scaler = preprocess_dual_path_data(
        raw_curves, feat_data, labels
    )
    
    # 划分训练测试集
    X_raw_train, X_raw_test, X_feat_train, X_feat_test, y_train, y_test = train_test_split(
        raw_curves_scaled, feat_data_scaled, labels, test_size=0.2, random_state=RANDOM_SEED, stratify=labels
    )
    
    # 创建数据集
    train_dataset = DualPathDataset(X_raw_train, X_feat_train, y_train)
    test_dataset = DualPathDataset(X_raw_test, X_feat_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    
    # 创建模型
    num_classes = len(np.unique(labels))  # 使用实际类别数
    model = HybridXGBoostV11(
        feat_in_dim=feat_data.shape[1],
        num_classes=num_classes
    ).to(device)
    
    print(f"模型参数: {sum(p.numel() for p in model.parameters())}")
    
    # 创建优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    # 训练XGBoost分类器（使用训练数据）
    print("训练XGBoost分类器...")
    train_raw_tensor = torch.tensor(X_raw_train, dtype=torch.float32).to(device)
    train_feat_tensor = torch.tensor(X_feat_train, dtype=torch.float32).to(device)
    train_labels_tensor = torch.tensor(y_train, dtype=torch.long).to(device)
    
    model.fit_xgb(train_raw_tensor, train_feat_tensor, train_labels_tensor)
    print("XGBoost分类器训练完成！")
    
    # 训练循环
    print(f"开始训练 {epochs} 个epoch...")
    best_f1 = 0.0
    best_epoch = 0
    training_log = []
    
    for epoch in range(epochs):
        # 训练
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, device, xgb_weight)
        
        # 评估
        test_results = evaluate_model(model, test_loader, device)
        
        # 更新学习率
        scheduler.step()
        
        # 记录日志
        training_log.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'test_loss': test_results['loss'],
            'test_acc': test_results['accuracy'],
            'test_f1': test_results['f1'],
            'lr': optimizer.param_groups[0]['lr']
        })
        
        # 保存最佳模型
        if test_results['f1'] > best_f1:
            best_f1 = test_results['f1']
            best_epoch = epoch + 1
            out_dir.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), out_dir / 'best_model.pt')
        
        # 打印进度
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d}/{epochs} | "
                  f"Train: loss={train_loss:.4f} acc={train_acc:.4f} | "
                  f"Test: acc={test_results['accuracy']:.4f} f1={test_results['f1']:.4f} | "
                  f"LR: {optimizer.param_groups[0]['lr']:.6f}")
    
    print(f"\n训练完成！最佳测试F1: {best_f1:.4f} (Epoch {best_epoch})")
    
    # 加载最佳模型进行最终评估
    print("加载最佳模型进行最终评估...")
    model.load_state_dict(torch.load(out_dir / 'best_model.pt'))
    
    # 最终评估
    final_results = evaluate_model(model, test_loader, device)
    
    print(f"\n=== 最终测试结果 ===")
    print(f"准确率: {final_results['accuracy']:.4f}")
    print(f"精确率: {final_results['precision']:.4f}")
    print(f"召回率: {final_results['recall']:.4f}")
    print(f"F1分数: {final_results['f1']:.4f}")
    
    # 保存结果
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存训练日志
    log_df = pd.DataFrame(training_log)
    log_df.to_csv(out_dir / 'training_log.csv', index=False)
    
    # 保存测试指标
    test_metrics = pd.DataFrame([{
        'size': size_tag,
        'model': 'hybrid_xgboost_v11',
        'best_epoch': best_epoch,
        'best_f1': best_f1,
        'final_accuracy': final_results['accuracy'],
        'final_precision': final_results['precision'],
        'final_recall': final_results['recall'],
        'final_f1': final_results['f1'],
        'epochs': epochs,
        'batch_size': batch_size,
        'learning_rate': lr,
        'xgb_weight': xgb_weight
    }])
    test_metrics.to_csv(out_dir / 'test_metrics.csv', index=False)
    
    # 保存混淆矩阵
    cm_df = pd.DataFrame(
        final_results['confusion_matrix'],
        index=[f'True_{i}' for i in range(len(final_results['confusion_matrix']))],
        columns=[f'Pred_{i}' for i in range(len(final_results['confusion_matrix']))]
    )
    cm_df.to_csv(out_dir / 'confusion_matrix.csv')
    
    # 保存模型
    torch.save({
        'model_state_dict': model.state_dict(),
        'raw_scaler': raw_scaler,
        'feat_scaler': feat_scaler,
        'xgb_classifier': model.xgb_loss.xgb_classifier,
        'label_encoder': model.xgb_loss.label_encoder
    }, out_dir / 'model.pt')
    
    print(f"结果已保存到: {out_dir}")


def main():
    parser = argparse.ArgumentParser(description='Hybrid XGBoost v11 Training')
    parser.add_argument('--size', type=str, required=True, choices=['035', '05'], help='Dataset size')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--xgb_weight', type=float, default=0.5, help='XGBoost loss weight')
    parser.add_argument('--out', type=str, default='experiments_single', help='Output directory')
    
    args = parser.parse_args()
    
    set_seed()
    
    out_dir = Path(args.out) / 'v11' / args.size
    run(
        size_tag=args.size,
        out_dir=out_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        xgb_weight=args.xgb_weight
    )


if __name__ == '__main__':
    main()
