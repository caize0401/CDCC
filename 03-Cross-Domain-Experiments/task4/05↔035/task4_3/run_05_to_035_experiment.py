"""
跨域实验：0.5数据集训练，0.35数据集测试
使用task3的两个混合模型
"""
import os
import random
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler

from common_utils import (
    load_datasets, align_and_prepare, fit_scaler_on_union, 
    encode_labels_union, HybridDataset, train_one_epoch, 
    evaluate, save_results
)
from hybrid_models import HybridFusion, AdvancedHybridFusion


RANDOM_SEED = 42


def set_seed(seed: int = RANDOM_SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def run_hybrid_experiment(
    model_class, model_name: str, 
    X_raw_train: np.ndarray, X_feat_train: np.ndarray, y_train: np.ndarray,
    X_raw_test: np.ndarray, X_feat_test: np.ndarray, y_test: np.ndarray,
    feat_cols: List[str], output_dir: Path, le,
    epochs: int = 30, batch_size: int = 64, lr: float = 1e-3
) -> Dict[str, float]:
    """运行单个混合模型的跨域实验"""
    
    # 标准化
    raw_scaler = StandardScaler().fit(X_raw_train)
    feat_scaler = StandardScaler().fit(X_feat_train)
    
    # 创建数据集
    train_ds = HybridDataset(X_raw_train, X_feat_train, y_train, raw_scaler, feat_scaler)
    test_ds = HybridDataset(X_raw_test, X_feat_test, y_test, raw_scaler, feat_scaler)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    
    # 创建模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_classes = int(np.max(np.concatenate([y_train, y_test])) + 1)
    
    if model_class == HybridFusion:
        model = HybridFusion(
            feat_in_dim=X_feat_train.shape[1], 
            num_classes=num_classes,
            label_smoothing=0.1
        ).to(device)
    else:  # AdvancedHybridFusion
        model = AdvancedHybridFusion(
            feat_in_dim=X_feat_train.shape[1], 
            num_classes=num_classes,
            label_smoothing=0.1
        ).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    
    best_val_acc = 0.0
    
    # 训练循环
    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, device)
        val_loss, val_acc, _, _ = evaluate(model, test_loader, device)
        scheduler.step(val_loss)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
        
        if epoch % 5 == 0:
            print(f"[{model_name}] Epoch {epoch}/{epochs} | "
                  f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
                  f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} best={best_val_acc:.4f}")
    
    # 最终测试
    _, _, y_pred, y_true = evaluate(model, test_loader, device)
    
    # 保存结果
    metrics = save_results(y_true, y_pred, model_name, "cross_domain", output_dir, le)
    
    return metrics


def run_all_experiments():
    """运行所有跨域实验"""
    set_seed()
    
    base_dir = Path(__file__).resolve().parent
    output_dir = Path(__file__).resolve().parent / '05_to_035'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 加载数据集
    datasets = load_datasets(base_dir)
    
    all_results = []
    
    # 混合模型实验：使用data1(原始曲线) + data2(完整特征)
    print(f"\n=== 混合模型跨域实验：0.5→0.35 ===")
    
    # 训练集：0.5，测试集：0.35
    df_train_raw = datasets['data1']['05']    # 原始信号
    df_train_feat = datasets['data2']['05']   # 完整特征
    df_test_raw = datasets['data1']['035']    # 原始信号
    df_test_feat = datasets['data2']['035']   # 完整特征
    
    # 对齐和准备数据
    X_raw_train, X_feat_train, y_train, feat_cols = align_and_prepare(df_train_raw, df_train_feat)
    X_raw_test, X_feat_test, y_test, _ = align_and_prepare(df_test_raw, df_test_feat)
    
    # 标签编码
    y_train_enc, y_test_enc, le = encode_labels_union(y_train, y_test)
    
    print(f"训练集大小: {len(y_train_enc)}, 测试集大小: {len(y_test_enc)}")
    print(f"原始信号维度: {X_raw_train.shape[1]}, 特征维度: {X_feat_train.shape[1]}, 类别数: {len(le.classes_)}")
    
    # 运行HybridFusion实验
    print(f"\n--- 运行 HybridFusion 实验 ---")
    metrics_v1 = run_hybrid_experiment(
        HybridFusion, "HybridFusion",
        X_raw_train, X_feat_train, y_train_enc,
        X_raw_test, X_feat_test, y_test_enc,
        feat_cols, output_dir, le
    )
    all_results.append({
        'model': 'HybridFusion', 
        'direction': '05_to_035',
        **metrics_v1
    })
    
    # 运行AdvancedHybridFusion实验
    print(f"\n--- 运行 AdvancedHybridFusion 实验 ---")
    metrics_v2 = run_hybrid_experiment(
        AdvancedHybridFusion, "AdvancedHybridFusion",
        X_raw_train, X_feat_train, y_train_enc,
        X_raw_test, X_feat_test, y_test_enc,
        feat_cols, output_dir, le
    )
    all_results.append({
        'model': 'AdvancedHybridFusion', 
        'direction': '05_to_035',
        **metrics_v2
    })
    
    # 保存汇总结果
    summary_df = pd.DataFrame(all_results)
    summary_df.to_csv(output_dir / 'summary.csv', index=False, encoding='utf-8-sig')
    
    print(f"\n=== 实验完成 ===")
    print(f"结果保存在: {output_dir}")
    print("\n汇总结果:")
    print(summary_df.to_string(index=False))


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    run_all_experiments()
