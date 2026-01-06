"""
跨域实验：0.5数据集训练，0.35数据集测试
使用v4 (Transformer) 和 v5 (HybridFusion Pro) 模型
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
    load_datasets, prepare_transformer_data, align_and_prepare_hybrid_data,
    encode_labels_union, TransformerDataset, HybridDataset,
    get_transformer_model, get_hybrid_model, _extract_logits, 
    accuracy, save_results
)


RANDOM_SEED = 42


def set_seed(seed: int = RANDOM_SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_one_epoch_transformer(model, loader, optimizer, device):
    """训练Transformer模型一个epoch"""
    model.train()
    total_loss = 0.0
    total_acc = 0.0
    total_n = 0
    
    for raw, y in loader:
        raw = raw.to(device)  # (batch_size, seq_len)
        y = y.to(device)
        
        optimizer.zero_grad(set_to_none=True)
        
        # 前向传播
        outputs = model(raw)
        
        # 计算损失
        if hasattr(model, 'loss'):
            try:
                loss = model.loss(outputs, y)
            except TypeError:
                logits = _extract_logits(outputs)
                loss = model.loss(logits, y)
        else:
            logits = _extract_logits(outputs)
            loss = nn.functional.cross_entropy(logits, y)
        
        # 反向传播
        loss.backward()
        
        # 梯度裁剪（对Transformer很重要）
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # 统计
        batch = y.size(0)
        total_loss += loss.item() * batch
        logits_for_acc = _extract_logits(outputs)
        total_acc += (logits_for_acc.argmax(dim=1) == y).float().sum().item()
        total_n += batch
    
    return total_loss / total_n, total_acc / total_n


def train_one_epoch_hybrid(model, loader, optimizer, device):
    """训练混合模型一个epoch"""
    model.train()
    total_loss = 0.0
    total_acc = 0.0
    total_n = 0
    
    for raw, feat, y in loader:
        raw = raw.to(device)
        feat = feat.to(device)
        y = y.to(device)

        optimizer.zero_grad(set_to_none=True)
        
        # 前向传播
        outputs = model(raw, feat)
        
        # 计算损失
        if hasattr(model, 'loss'):
            try:
                loss = model.loss(outputs, y)
            except TypeError:
                logits = _extract_logits(outputs)
                loss = model.loss(logits, y)
        else:
            logits = _extract_logits(outputs)
            loss = nn.functional.cross_entropy(logits, y)
        
        loss.backward()
        optimizer.step()

        batch = y.size(0)
        total_loss += loss.item() * batch
        logits_for_acc = _extract_logits(outputs)
        total_acc += accuracy(logits_for_acc, y) * batch
        total_n += batch
    return total_loss / total_n, total_acc / total_n


@torch.no_grad()
def evaluate_transformer(model, loader, device):
    """评估Transformer模型"""
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    total_n = 0
    all_preds = []
    all_targets = []
    
    for raw, y in loader:
        raw = raw.to(device)
        y = y.to(device)
        
        outputs = model(raw)
        
        # 计算损失
        if hasattr(model, 'loss'):
            try:
                loss = model.loss(outputs, y)
            except TypeError:
                logits = _extract_logits(outputs)
                loss = model.loss(logits, y)
        else:
            logits = _extract_logits(outputs)
            loss = nn.functional.cross_entropy(logits, y)
        
        batch = y.size(0)
        total_loss += loss.item() * batch
        logits_for_acc = _extract_logits(outputs)
        total_acc += (logits_for_acc.argmax(dim=1) == y).float().sum().item()
        total_n += batch
        
        preds = logits_for_acc.argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_targets.extend(y.cpu().numpy())
    
    return total_loss / total_n, total_acc / total_n, np.array(all_preds), np.array(all_targets)


@torch.no_grad()
def evaluate_hybrid(model, loader, device):
    """评估混合模型"""
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    total_n = 0
    all_preds = []
    all_targets = []
    
    for raw, feat, y in loader:
        raw = raw.to(device)
        feat = feat.to(device)
        y = y.to(device)

        outputs = model(raw, feat)
        
        # 推理时只返回主logits
        logits = _extract_logits(outputs)
        loss = nn.functional.cross_entropy(logits, y)

        batch = y.size(0)
        total_loss += loss.item() * batch
        total_acc += accuracy(logits, y) * batch
        total_n += batch
        
        preds = logits.argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_targets.extend(y.cpu().numpy())
        
    return total_loss / total_n, total_acc / total_n, np.array(all_preds), np.array(all_targets)


def run_transformer_experiment(
    model_name: str,
    X_raw_train: np.ndarray, y_train: np.ndarray,
    X_raw_test: np.ndarray, y_test: np.ndarray,
    output_dir: Path, le,
    epochs: int = 30, batch_size: int = 32, lr: float = 1e-4
) -> Dict[str, float]:
    """运行Transformer v4实验"""
    
    # 标准化
    scaler = StandardScaler()
    X_train_flat = X_raw_train.reshape(-1, X_raw_train.shape[-1])
    scaler.fit(X_train_flat)
    
    # 创建数据集
    train_ds = TransformerDataset(X_raw_train, y_train, scaler)
    test_ds = TransformerDataset(X_raw_test, y_test, scaler)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    
    # 创建模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_classes = len(np.unique(np.concatenate([y_train, y_test])))
    input_dim = X_raw_train.shape[1]  # 序列长度
    
    model = get_transformer_model(model_name, input_dim=input_dim, num_classes=num_classes)
    model = model.to(device)
    
    # 优化器和调度器
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    
    # Warmup + Cosine调度器
    def lr_lambda(epoch):
        warmup_epochs = 5
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        else:
            if epochs == warmup_epochs:
                return 1.0
            return 0.5 * (1 + np.cos(np.pi * (epoch - warmup_epochs) / (epochs - warmup_epochs)))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    best_val_acc = 0.0
    
    # 训练循环
    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch_transformer(model, train_loader, optimizer, device)
        val_loss, val_acc, _, _ = evaluate_transformer(model, test_loader, device)
        scheduler.step()
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
        
        if epoch % 5 == 0:
            print(f"[{model_name}] Epoch {epoch}/{epochs} | "
                  f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
                  f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} best={best_val_acc:.4f}")
    
    # 最终测试
    _, _, y_pred, y_true = evaluate_transformer(model, test_loader, device)
    
    # 保存结果
    metrics = save_results(y_true, y_pred, model_name, "cross_domain", output_dir, le)
    
    return metrics


def run_hybrid_experiment(
    model_name: str,
    X_raw_train: np.ndarray, X_feat_train: np.ndarray, y_train: np.ndarray,
    X_raw_test: np.ndarray, X_feat_test: np.ndarray, y_test: np.ndarray,
    feat_cols: List[str], output_dir: Path, le,
    epochs: int = 30, batch_size: int = 64, lr: float = 1e-3
) -> Dict[str, float]:
    """运行混合模型v5实验"""
    
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
    num_classes = len(np.unique(np.concatenate([y_train, y_test])))
    
    model = get_hybrid_model(model_name, feat_in_dim=X_feat_train.shape[1], num_classes=num_classes)
    model = model.to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    
    best_val_acc = 0.0
    
    # 训练循环
    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch_hybrid(model, train_loader, optimizer, device)
        val_loss, val_acc, _, _ = evaluate_hybrid(model, test_loader, device)
        scheduler.step(val_loss)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
        
        if epoch % 5 == 0:
            print(f"[{model_name}] Epoch {epoch}/{epochs} | "
                  f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
                  f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} best={best_val_acc:.4f}")
    
    # 最终测试
    _, _, y_pred, y_true = evaluate_hybrid(model, test_loader, device)
    
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
    
    print(f"\n=== 跨域实验：0.5→0.35 ===")
    
    # ==================== Transformer v4 实验 ====================
    print(f"\n--- 运行 Transformer v4 实验 ---")
    
    # 准备Transformer数据（只需要data1）
    df_train_raw = datasets['data1']['05']    # 训练：0.5
    df_test_raw = datasets['data1']['035']    # 测试：0.35
    
    X_raw_train, y_train = prepare_transformer_data(df_train_raw)
    X_raw_test, y_test = prepare_transformer_data(df_test_raw)
    
    # 标签编码
    y_train_enc, y_test_enc, le_transformer = encode_labels_union(y_train, y_test)
    
    print(f"Transformer - 训练集大小: {len(y_train_enc)}, 测试集大小: {len(y_test_enc)}")
    print(f"Transformer - 序列长度: {X_raw_train.shape[1]}, 类别数: {len(le_transformer.classes_)}")
    
    # 运行Transformer实验
    metrics_v4 = run_transformer_experiment(
        "transformer_v4",
        X_raw_train, y_train_enc,
        X_raw_test, y_test_enc,
        output_dir, le_transformer
    )
    all_results.append({
        'model': 'TransformerV4', 
        'direction': '05_to_035',
        **metrics_v4
    })
    
    # ==================== HybridFusion v5 实验 ====================
    print(f"\n--- 运行 HybridFusion v5 实验 ---")
    
    # 准备混合模型数据（需要data1+data2）
    df_train_raw = datasets['data1']['05']    # 原始信号
    df_train_feat = datasets['data2']['05']   # 完整特征
    df_test_raw = datasets['data1']['035']    # 原始信号
    df_test_feat = datasets['data2']['035']   # 完整特征
    
    # 对齐和准备数据
    X_raw_train, X_feat_train, y_train, feat_cols = align_and_prepare_hybrid_data(df_train_raw, df_train_feat)
    X_raw_test, X_feat_test, y_test, _ = align_and_prepare_hybrid_data(df_test_raw, df_test_feat)
    
    # 标签编码
    y_train_enc, y_test_enc, le_hybrid = encode_labels_union(y_train, y_test)
    
    print(f"混合模型 - 训练集大小: {len(y_train_enc)}, 测试集大小: {len(y_test_enc)}")
    print(f"混合模型 - 原始信号维度: {X_raw_train.shape[1]}, 特征维度: {X_feat_train.shape[1]}, 类别数: {len(le_hybrid.classes_)}")
    
    # 运行混合模型实验
    metrics_v5 = run_hybrid_experiment(
        "hybrid_fusion_v5",
        X_raw_train, X_feat_train, y_train_enc,
        X_raw_test, X_feat_test, y_test_enc,
        feat_cols, output_dir, le_hybrid
    )
    all_results.append({
        'model': 'HybridFusionV5', 
        'direction': '05_to_035',
        **metrics_v5
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
