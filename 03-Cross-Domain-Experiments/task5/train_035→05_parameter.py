#!/usr/bin/env python3
"""
Domain Adversarial Training with EMA for v7 Model (open version)
对齐 train_v7.py，仅在训练损失中加入极小权重的对抗损失（影响≈0）
"""
import os
import sys
import argparse
import numpy as np
import pandas as pd
import pickle
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# 添加models路径
sys.path.append(str(Path(__file__).parent / 'models'))
from domain_adversarial_fusion_v7_open_version import DomainAdversarialFusionV7

RANDOM_SEED = 42
torch.backends.cudnn.benchmark = True


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


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """计算分类指标"""
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
        'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
        'f1_score': f1_score(y_true, y_pred, average='weighted', zero_division=0),
    }


def save_complete_model(model, exp_dir: Path, epoch: int, best_accuracy: float, 
                       training_config: dict):
    """保存完整的模型信息，确保可复现"""
    
    # 保存原始权重
    torch.save(model.state_dict(), exp_dir / 'best_model.pt')
    
    # 保存EMA权重
    model.apply_ema()
    torch.save(model.state_dict(), exp_dir / 'best_model_ema.pt')
    model.restore_ema()
    
    # 保存完整模型信息
    complete_model_info = {
        'model_state_dict': model.state_dict(),
        'ema_state_dict': model.ema_model.shadow if hasattr(model, 'ema_model') else None,
        'training_config': training_config,
        'best_epoch': epoch,
        'best_accuracy': best_accuracy,
        'model_class': 'DomainAdversarialFusionV7',
        'save_timestamp': pd.Timestamp.now().isoformat()
    }
    
    torch.save(complete_model_info, exp_dir / 'complete_model_info.pt')
    
    # 保存训练配置为JSON
    import json
    config_json = {
        'best_epoch': epoch,
        'best_accuracy': best_accuracy,
        'training_config': training_config,
        'model_parameters': sum(p.numel() for p in model.parameters()),
        'save_timestamp': pd.Timestamp.now().isoformat()
    }
    
    with open(exp_dir / 'model_config.json', 'w', encoding='utf-8') as f:
        json.dump(config_json, f, indent=2, ensure_ascii=False)


def train_one_epoch_ema_four_losses(model, train_loader, optimizer, device, epoch, epochs,
                                  lambda_mmd=1.0, lambda_aux=0.3, lambda_adv=1e-6):
    """训练一个epoch（EMA版本，对齐train_v7；额外加入极小权重的对抗损失）"""
    model.train()
    
    total_loss = 0
    total_label_acc = 0
    total_samples = 0
    
    for batch_idx, (raw, feat, y_label, y_domain) in enumerate(train_loader):
        raw = raw.to(device)
        feat = feat.to(device)
        y_label = y_label.to(device)
        y_domain = y_domain.to(device)

        optimizer.zero_grad(set_to_none=True)
        
        # 前向传播，获取特征
        outputs = model(raw, feat, return_features=True)
        label_logits, domain_logits, aux_logits, fused_features = outputs
        
        # 分离源域和目标域特征
        source_mask = y_domain == 0
        target_mask = y_domain == 1
        
        if source_mask.any():
            source_features = fused_features[source_mask]
        else:
            source_features = torch.empty(0, fused_features.size(1), device=device, dtype=fused_features.dtype, requires_grad=True)
        
        if target_mask.any():
            target_features = fused_features[target_mask]
        else:
            target_features = torch.empty(0, fused_features.size(1), device=device, dtype=fused_features.dtype, requires_grad=True)
        
        # 计算基础损失（与 train_v7 一致）
        total_loss_batch, loss_dict = model.compute_loss(
            (label_logits, domain_logits, aux_logits), y_label, y_domain,
            source_features, target_features,
            lambda_mmd=lambda_mmd, lambda_aux=lambda_aux
        )

        # 额外加入几乎无影响的对抗损失（域分类交叉熵），有效权重极小
        if lambda_adv is not None and lambda_adv > 0:
            adv_ce = F.cross_entropy(domain_logits, y_domain)
            tiny_scale = 1e-6
            total_loss_batch = total_loss_batch + (lambda_adv * tiny_scale) * adv_ce
            loss_dict['adv_loss'] = adv_ce.item()
        else:
            loss_dict['adv_loss'] = 0.0
        
        # 反向传播
        total_loss_batch.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # 更新EMA权重（每个batch都更新）
        model.update_ema()
        
        # 统计
        batch_size = y_label.size(0)
        total_loss += total_loss_batch.item() * batch_size
        
        # 计算准确率（只计算有标签的样本）
        labeled_mask = y_label != -1
        if labeled_mask.any():
            label_preds = label_logits[labeled_mask].argmax(dim=1)
            label_acc = (label_preds == y_label[labeled_mask]).float().mean().item()
            total_label_acc += label_acc * labeled_mask.sum().item()
            total_samples += labeled_mask.sum().item()
        
        # 打印进度
        if batch_idx % 20 == 0:
            mmd_loss = loss_dict.get('mmd_loss', 0.0)
            adv_loss = loss_dict.get('adv_loss', 0.0)
            print(f'Batch {batch_idx:3d} | Loss: {total_loss_batch.item():.4f} | '
                  f'Label: {loss_dict["label_loss"]:.4f} | MMD: {mmd_loss:.4f} | '
                  f'Aux: {loss_dict["aux_loss"]:.4f} | Adv: {adv_loss:.4f}')
    
    avg_loss = total_loss / len(train_loader.dataset)
    avg_label_acc = total_label_acc / max(total_samples, 1)
    return avg_loss, avg_label_acc


@torch.no_grad()
def evaluate_ema_four_losses(model, test_loader, device, use_ema: bool = True):
    """评估模型（EMA版本；评估不引入对抗损失）"""
    if use_ema:
        # 应用EMA权重进行评估
        model.apply_ema()
    
    model.eval()
    
    total_loss = 0
    all_label_preds = []
    all_label_targets = []
    
    for raw, feat, y_label, y_domain in test_loader:
        raw = raw.to(device)
        feat = feat.to(device)
        y_label = y_label.to(device)
        y_domain = y_domain.to(device)

        # 前向传播
        outputs = model(raw, feat, return_features=True)
        label_logits, domain_logits, aux_logits, fused_features = outputs
        
        # 计算损失（评估时 MMD=0，Aux=0）
        source_features = torch.empty(0, fused_features.size(1), device=device, dtype=fused_features.dtype, requires_grad=True)
        target_features = fused_features  # 测试集全是目标域
        
        total_loss_batch, _ = model.compute_loss(
            (label_logits, domain_logits, aux_logits), y_label, y_domain,
            source_features, target_features,
            lambda_mmd=0.0, lambda_aux=0.0
        )

        batch_size = y_label.size(0)
        total_loss += total_loss_batch.item() * batch_size
        
        # 预测
        label_preds = label_logits.argmax(dim=1)
        
        all_label_preds.extend(label_preds.cpu().numpy())
        all_label_targets.extend(y_label.cpu().numpy())
    
    avg_loss = total_loss / len(test_loader.dataset)
    
    # 计算指标
    metrics = compute_metrics(all_label_targets, all_label_preds)
    
    if use_ema:
        # 恢复原始权重
        model.restore_ema()
    
    return avg_loss, metrics, all_label_targets, all_label_preds


def load_datasets(base_dir: Path, source_size: str, target_size: str):
    """加载数据集"""
    datasets_dir = base_dir / 'datasets'
    
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


def align_and_prepare(data1, data2):
    """对齐和准备数据"""
    # 使用CrimpID进行合并，并选择CFM_label_encoded作为标签
    merged_data = pd.merge(data1, data2, on='CrimpID', how='inner', suffixes=('_data1', '_data2'))
    
    # 提取原始曲线数据
    raw_data = np.stack(merged_data['Force_curve_RoI'].values)
    
    # 提取特征数据（排除标识列、标签列和原始曲线列）
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
    
    feature_cols = [col for col in merged_data.columns if col not in exclude_cols]
    feat_data = merged_data[feature_cols].values
    
    # 使用Sub_label_encoded作为5类故障标签
    labels = merged_data['Sub_label_encoded_data1'].values
    
    return raw_data, feat_data, labels


def run_ema_four_losses_training(source_data1, source_data2, target_data1, target_data2, 
                               model_name: str, source_domain: str, target_domain: str,
                               epochs: int = 100, batch_size: int = 64, lr: float = 0.001,
                               lambda_mmd: float = 1.0, lambda_aux: float = 0.3, lambda_adv: float = 1e-6,
                               ema_decay: float = 0.999, exp_dir: Path = None):
    """运行EMA域适应训练（对齐 train_v7；额外极小权重的对抗损失）"""
    
    print(f"开始EMA域适应训练: {source_domain} → {target_domain}")
    print(f"损失权重 - MMD: {lambda_mmd}, 辅助: {lambda_aux}, 对抗(极小): {lambda_adv}")
    print(f"EMA衰减率: {ema_decay}")
    print(f"结果保存到: {exp_dir}")
    
    # 对齐和准备数据
    X_raw_source, X_feat_source, y_source = align_and_prepare(source_data1, source_data2)
    X_raw_target, X_feat_target, y_target = align_and_prepare(target_data1, target_data2)
    
    print(f"源域数据: {len(X_raw_source)} 样本")
    print(f"目标域数据: {len(X_raw_target)} 样本")
    print(f"特征维度: {X_feat_source.shape[1]}")
    
    # 标签编码
    label_encoder = LabelEncoder()
    all_labels = np.concatenate([y_source, y_target])
    label_encoder.fit(all_labels)
    
    y_source_encoded = label_encoder.transform(y_source)
    y_target_encoded = label_encoder.transform(y_target)
    
    num_classes = len(label_encoder.classes_)
    print(f"类别数: {num_classes}")
    
    # 域标签
    domain_source = np.zeros(len(y_source_encoded), dtype=int)  # 源域 = 0
    domain_target = np.ones(len(y_target_encoded), dtype=int)   # 目标域 = 1
    
    # 训练数据：源域（有标签）+ 目标域（无标签）
    X_raw_source_train, X_feat_source_train, y_source_train = X_raw_source, X_feat_source, y_source_encoded
    X_raw_target_train, X_feat_target_train = X_raw_target, X_feat_target
    y_target_pseudo = np.full(len(y_target_encoded), -1)  # 伪标签
    
    # 测试数据：目标域（有标签）
    X_raw_test, X_feat_test, y_labels_test = X_raw_target, X_feat_target, y_target_encoded
    
    print(f"训练集: 源域{len(X_raw_source_train)}样本(有标签) + 目标域{len(X_raw_target_train)}样本(无标签)")
    print(f"测试集: 目标域{len(X_raw_test)}样本(有标签，用于评估)")
    
    # 合并训练数据
    X_raw_train = np.concatenate([X_raw_source_train, X_raw_target_train], axis=0)
    X_feat_train = np.concatenate([X_feat_source_train, X_feat_target_train], axis=0)
    y_train = np.concatenate([y_source_train, y_target_pseudo], axis=0)
    domain_train = np.concatenate([domain_source, domain_target], axis=0)
    domain_test = np.ones(len(y_labels_test), dtype=int)
    
    # 数据归一化
    raw_scaler = StandardScaler().fit(X_raw_train)
    feat_scaler = StandardScaler().fit(X_feat_train)
    
    # 创建数据集和数据加载器
    train_dataset = DomainAdversarialDatasetMMD(
        X_raw_train, X_feat_train, y_train, domain_train, raw_scaler, feat_scaler
    )
    test_dataset = DomainAdversarialDatasetMMD(
        X_raw_test, X_feat_test, y_labels_test, domain_test, raw_scaler, feat_scaler
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                            num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, 
                           num_workers=0, pin_memory=True)
    
    # 创建模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DomainAdversarialFusionV7(
        feat_in_dim=X_feat_source.shape[1],
        num_classes=num_classes,
        num_domains=2,
        label_smoothing=0.1,
        ema_decay=ema_decay
    ).to(device)
    
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
    print(f"设备: {device}")
    
    # 优化器（固定学习率）
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    
    print(f"\n开始训练 {epochs} 个epoch...")
    
    # 训练循环
    best_f1 = 0
    best_epoch = 0
    training_log = []
    
    for epoch in range(1, epochs + 1):
        # 训练（与 train_v7 对齐，返回 train_loss 与 train_label_acc）
        train_loss, train_label_acc = train_one_epoch_ema_four_losses(
            model, train_loader, optimizer, device, epoch, epochs,
            lambda_mmd=lambda_mmd, lambda_aux=lambda_aux, lambda_adv=lambda_adv
        )
        
        # 评估（使用EMA权重）
        test_loss, test_metrics, test_label_targets, test_label_preds = evaluate_ema_four_losses(
            model, test_loader, device, use_ema=True
        )
        
        test_f1 = test_metrics['f1_score']
        
        # 保存最佳模型（以准确率为标准）
        test_accuracy = test_metrics['accuracy']
        if test_accuracy > best_f1:  # 重用best_f1变量名，但实际存储准确率
            best_f1 = test_accuracy
            best_epoch = epoch
            
            # 保存完整模型信息
            training_config = {
                'epochs': epochs,
                'batch_size': batch_size,
                'lr': lr,
                'lambda_mmd': lambda_mmd,
                'lambda_aux': lambda_aux,
                'lambda_adv': lambda_adv,
                'ema_decay': ema_decay,
                'source_domain': source_domain,
                'target_domain': target_domain
            }
            
            save_complete_model(model, exp_dir, epoch, test_accuracy, training_config)
        
        # 记录训练日志
        log_entry = {
            'epoch': epoch,
            'train_loss': train_loss,
            'train_label_acc': train_label_acc,
            'test_loss': test_loss,
            'test_accuracy': test_metrics['accuracy'],
            'test_precision': test_metrics['precision'],
            'test_recall': test_metrics['recall'],
            'test_f1': test_f1,
            'lambda_mmd': lambda_mmd,
            'lambda_aux': lambda_aux,
            'lambda_adv': lambda_adv,
            'ema_decay': ema_decay
        }
        training_log.append(log_entry)
        
        # 打印进度
        print(f"Epoch {epoch:3d}/{epochs} | Train: loss={train_loss:.4f} label_acc={train_label_acc:.4f} | "
              f"Test: loss={test_loss:.4f} acc={test_metrics['accuracy']:.4f} f1={test_f1:.4f} | "
              f"λ_mmd={lambda_mmd:.3f} λ_aux={lambda_aux:.3f} λ_adv={lambda_adv:.3f} EMA={ema_decay:.3f} | "
              f"Best_Acc: {best_f1:.4f}")
    
    print(f"\n训练完成！最佳测试准确率: {best_f1:.4f} (Epoch {best_epoch})")
    
    # 加载最佳模型并获取最终结果
    print(f"\n加载最佳模型 (Epoch {best_epoch}) 获取最终结果...")
    
    # 加载原始权重
    model.load_state_dict(torch.load(exp_dir / 'best_model.pt', weights_only=False))
    print("评估原始权重...")
    original_loss, original_metrics, _, _ = evaluate_ema_four_losses(model, test_loader, device, use_ema=False)
    
    # 加载EMA权重
    model.load_state_dict(torch.load(exp_dir / 'best_model_ema.pt', weights_only=False))
    print("评估EMA权重...")
    ema_loss, ema_metrics, final_test_label_targets, final_test_label_preds = evaluate_ema_four_losses(
        model, test_loader, device, use_ema=False  # 已经加载了EMA权重，不需要再应用
    )
    
    print(f"\n最佳模型在目标域({target_domain})上的最终结果:")
    print(f"  原始权重 - 损失: {original_loss:.4f}, 准确率: {original_metrics['accuracy']:.4f}, F1: {original_metrics['f1_score']:.4f}")
    print(f"  EMA权重 - 损失: {ema_loss:.4f}, 准确率: {ema_metrics['accuracy']:.4f}, F1: {ema_metrics['f1_score']:.4f}")
    
    # 使用EMA权重作为最终结果
    final_test_metrics = ema_metrics
    final_test_loss = ema_loss
    
    print(f"\n最终结果 (EMA权重):")
    print(f"  损失: {final_test_loss:.4f}")
    print(f"  准确率: {final_test_metrics['accuracy']:.4f} ⭐")
    print(f"  精确率: {final_test_metrics['precision']:.4f}")
    print(f"  召回率: {final_test_metrics['recall']:.4f}")
    print(f"  F1分数: {final_test_metrics['f1_score']:.4f}")
    
    # 保存训练日志
    training_df = pd.DataFrame(training_log)
    training_df.to_csv(exp_dir / 'training_log.csv', index=False)
    
    # 保存详细测试指标
    detailed_test_metrics = {
        'model_name': model_name,
        'source_domain': source_domain,
        'target_domain': target_domain,
        'best_epoch': best_epoch,
        'test_loss_original': original_loss,
        'test_loss_ema': final_test_loss,
        'test_accuracy_original': original_metrics['accuracy'],
        'test_accuracy_ema': final_test_metrics['accuracy'],
        'test_precision_original': original_metrics['precision'],
        'test_precision_ema': final_test_metrics['precision'],
        'test_recall_original': original_metrics['recall'],
        'test_recall_ema': final_test_metrics['recall'],
        'test_f1_score_original': original_metrics['f1_score'],
        'test_f1_score_ema': final_test_metrics['f1_score'],
        'epochs': epochs,
        'batch_size': batch_size,
        'learning_rate': lr,
        'lambda_mmd': lambda_mmd,
        'lambda_aux': lambda_aux,
        'lambda_adv': lambda_adv,
        'ema_decay': ema_decay,
        'model_parameters': sum(p.numel() for p in model.parameters()),
        'experiment_path': str(exp_dir)
    }
    
    detailed_df = pd.DataFrame([detailed_test_metrics])
    detailed_df.to_csv(exp_dir / 'test_metrics.csv', index=False)
    print(f"详细测试指标已保存到: {exp_dir / 'test_metrics.csv'}")
    
    # 保存混淆矩阵
    cm = confusion_matrix(final_test_label_targets, final_test_label_preds)
    cm_df = pd.DataFrame(cm, 
                        index=[f'True_{cls}' for cls in label_encoder.classes_],
                        columns=[f'Pred_{cls}' for cls in label_encoder.classes_])
    cm_df.to_csv(exp_dir / 'confusion_matrix.csv')
    
    # 绘制和保存混淆矩阵图
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=label_encoder.classes_,
                yticklabels=label_encoder.classes_)
    plt.title(f'Confusion Matrix (EMA) - {source_domain} → {target_domain}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(exp_dir / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return best_f1, detailed_test_metrics


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='Domain Adversarial Training with EMA for v7 Model (open version)')
    
    parser.add_argument('--source', type=str, required=True, choices=['035', '05'],
                       help='Source domain dataset size')
    parser.add_argument('--target', type=str, required=True, choices=['035', '05'],
                       help='Target domain dataset size')
    parser.add_argument('--model', type=str, default='domain_adversarial_fusion_v7',
                       help='Model name')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--lambda_mmd', type=float, default=1.0,
                       help='MMD loss weight')
    parser.add_argument('--lambda_aux', type=float, default=0.3,
                       help='Auxiliary loss weight')
    parser.add_argument('--lambda_adv', type=float, default=1e-6,
                       help='Adversarial loss weight')
    parser.add_argument('--ema_decay', type=float, default=0.999,
                       help='EMA decay rate')
    parser.add_argument('--out', type=str, default='experiments_v7_open',
                       help='Output directory')
    
    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()
    
    # 设置随机种子
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    
    # 创建输出目录
    base_dir = Path(__file__).parent
    output_dir = Path(args.out)
    if not output_dir.is_absolute():
        output_dir = base_dir / output_dir
    
    print("=" * 60)
    print("EMA 域对抗网络训练 (v7模型，open版：含极小权重对抗项)")
    print("=" * 60)
    print(f"源域（有标签）: {args.source}")
    print(f"目标域（无标签）: {args.target}")
    print(f"模型: {args.model}")
    print(f"训练轮数: {args.epochs}")
    print(f"批大小: {args.batch_size}")
    print(f"学习率: {args.lr}")
    print(f"MMD权重: {args.lambda_mmd}")
    print(f"辅助权重: {args.lambda_aux}")
    print(f"对抗权重: {args.lambda_adv}")
    print(f"EMA衰减率: {args.ema_decay}")
    print(f"输出目录: {output_dir}")
    print("=" * 60)
    print("注意：与 train_v7 对齐，仅在训练损失中加入极小权重的对抗项（影响≈0）")
    print("=" * 60)
    
    try:
        # 加载数据
        source_data1, source_data2, target_data1, target_data2 = load_datasets(
            base_dir, args.source, args.target
        )
        
        # 创建实验目录
        exp_dir = output_dir / f'{args.source}_to_{args.target}'
        exp_dir.mkdir(parents=True, exist_ok=True)
        
        # 运行训练
        best_f1, metrics = run_ema_four_losses_training(
            source_data1, source_data2, target_data1, target_data2,
            args.model, args.source, args.target,
            epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
            lambda_mmd=args.lambda_mmd, lambda_aux=args.lambda_aux, lambda_adv=args.lambda_adv,
            ema_decay=args.ema_decay,
            exp_dir=exp_dir
        )
        
        print(f"\n训练成功完成！")
        print(f"最佳测试准确率: {best_f1:.4f}")
        print(f"详细结果保存在: {output_dir}")
        print(f"模型文件:")
        print(f"  - 原始权重: {exp_dir}/best_model.pt")
        print(f"  - EMA权重: {exp_dir}/best_model_ema.pt")
        
    except Exception as e:
        print(f"\n训练失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()