import os
from pathlib import Path
from typing import Tuple, Dict, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


LABEL_COLS = ['CrimpID', 'Wire_cross-section_conductor', 'Main_label_string',
              'Sub_label_string', 'Main-label_encoded', 'Sub_label_encoded',
              'Binary_label_encoded', 'CFM_label_encoded']


def load_datasets(base_dir: Path) -> Dict[str, Dict[str, pd.DataFrame]]:
    """加载数据集"""
    src = base_dir / 'datasets'
    datasets = {
        'data1': {
            '035': pd.read_pickle(src / 'data1' / 'crimp_force_curves_dataset_035.pkl'),
            '05': pd.read_pickle(src / 'data1' / 'crimp_force_curves_dataset_05.pkl'),
        },
        'data2': {
            '035': pd.read_pickle(src / 'data2' / 'features_035.pkl'),
            '05': pd.read_pickle(src / 'data2' / 'features_05.pkl'),
        },
        'data3': {
            '035': pd.read_pickle(src / 'data3' / 'features_035_selected.pkl'),
            '05': pd.read_pickle(src / 'data3' / 'features_05_selected.pkl'),
        },
    }
    return datasets


def align_and_prepare(
    df_raw: pd.DataFrame, df_feat: pd.DataFrame
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """对齐原始信号数据和特征数据"""
    # Join on CrimpID to ensure alignment across paths
    left = df_raw[[
        'CrimpID', 'Force_curve_RoI', 'Sub_label_encoded'
    ]].copy()
    right = df_feat.copy()

    # 获取特征列（排除标签列和可能重复的列）
    feature_cols = [c for c in right.columns if c not in LABEL_COLS]
    
    # 先尝试基于CrimpID和Sub_label_encoded进行合并
    right_cols = ['CrimpID', 'Sub_label_encoded'] + feature_cols
    merged = pd.merge(left, right[right_cols], on=['CrimpID', 'Sub_label_encoded'], how='inner')

    if len(merged) == 0:
        print("警告：基于CrimpID和Sub_label_encoded的合并失败，尝试仅基于CrimpID合并")
        # 尝试仅基于CrimpID合并
        right_cols = ['CrimpID'] + feature_cols
        merged = pd.merge(left, right[right_cols], on='CrimpID', how='inner')
        
        if len(merged) == 0:
            print("警告：基于CrimpID的合并也失败，使用索引对齐")
            # Fallback: assume the order matches and lengths are the same
            min_len = min(len(left), len(right))
            left = left.iloc[:min_len].reset_index(drop=True)
            right = right.iloc[:min_len].reset_index(drop=True)
            merged = pd.concat([
                left[['CrimpID', 'Force_curve_RoI', 'Sub_label_encoded']],
                right[feature_cols]
            ], axis=1)
    
    # 更新特征列列表，排除可能在合并后不存在的列
    available_feature_cols = [c for c in feature_cols if c in merged.columns]

    print(f"合并后数据形状: {merged.shape}")
    
    # 检查Force_curve_RoI列是否存在，处理可能的列名冲突
    if 'Force_curve_RoI' in merged.columns:
        roi_col = 'Force_curve_RoI'
    elif 'Force_curve_RoI_x' in merged.columns:
        roi_col = 'Force_curve_RoI_x'  # 来自左表（原始信号数据）
    elif 'Force_curve_RoI_y' in merged.columns:
        roi_col = 'Force_curve_RoI_y'  # 来自右表
    else:
        raise ValueError(f"合并后的数据中缺少Force_curve_RoI相关列。可用列: {list(merged.columns)}")
    
    X_raw = np.stack(merged[roi_col].to_numpy())  # (N, 500)
    X_feat = merged[available_feature_cols].to_numpy(dtype=np.float32)  # (N, M)
    y = merged['Sub_label_encoded'].to_numpy(dtype=np.int64)

    # Ensure shapes
    if X_raw.ndim != 2:
        raise ValueError(f"Expected raw shape (N, L), got {X_raw.shape}")
    return X_raw.astype(np.float32), X_feat, y, available_feature_cols


def fit_scaler_on_union(X_train: np.ndarray, X_test: np.ndarray) -> StandardScaler:
    """在训练+测试数据的联合上拟合标准化器"""
    scaler = StandardScaler()
    scaler.fit(np.vstack([X_train, X_test]))
    return scaler


def encode_labels_union(y_train: np.ndarray, y_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray, LabelEncoder]:
    """在训练+测试标签的联合上拟合标签编码器"""
    le = LabelEncoder()
    y_union = np.concatenate([y_train, y_test])
    le.fit(y_union)
    return le.transform(y_train), le.transform(y_test), le


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """计算分类指标"""
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
        'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
        'f1_score': f1_score(y_true, y_pred, average='weighted', zero_division=0),
    }


class HybridDataset(Dataset):
    """混合模型的数据集类"""
    def __init__(
        self,
        X_raw: np.ndarray,
        X_feat: np.ndarray,
        y: np.ndarray,
        raw_scaler: StandardScaler,
        feat_scaler: StandardScaler,
    ) -> None:
        self.X_raw = X_raw
        self.X_feat = X_feat
        self.y = y
        self.raw_scaler = raw_scaler
        self.feat_scaler = feat_scaler

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        raw = self.X_raw[idx]
        feat = self.X_feat[idx]
        label = self.y[idx]

        # Normalize using prefit scalers
        raw_norm = self.raw_scaler.transform(raw.reshape(1, -1)).reshape(-1)
        feat_norm = self.feat_scaler.transform(feat.reshape(1, -1)).reshape(-1)

        # Convert to tensors; Conv1D expects (C, L)
        raw_tensor = torch.from_numpy(raw_norm).float().unsqueeze(0)
        feat_tensor = torch.from_numpy(feat_norm).float()
        label_tensor = torch.tensor(label, dtype=torch.long)
        return raw_tensor, feat_tensor, label_tensor


def accuracy(pred: torch.Tensor, target: torch.Tensor) -> float:
    """计算准确率"""
    preds = pred.argmax(dim=1)
    return (preds == target).float().mean().item()


def train_one_epoch(model, loader, optimizer, device):
    """训练一个epoch"""
    model.train()
    total_loss = 0.0
    total_acc = 0.0
    total_n = 0
    
    for raw, feat, y in loader:
        raw = raw.to(device)
        feat = feat.to(device)
        y = y.to(device)

        optimizer.zero_grad(set_to_none=True)
        
        # 处理不同模型的输出格式
        output = model(raw, feat)
        if isinstance(output, tuple):  # AdvancedHybridFusion在训练时返回tuple
            logits = output[0]
            loss = model.loss(output, y)
        else:  # HybridFusion返回单个logits
            logits = output
            loss = model.loss(logits, y)
            
        loss.backward()
        optimizer.step()

        batch = y.size(0)
        total_loss += loss.item() * batch
        total_acc += accuracy(logits.detach(), y) * batch
        total_n += batch
    return total_loss / total_n, total_acc / total_n


@torch.no_grad()
def evaluate(model, loader, device):
    """评估模型"""
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

        # 推理时只返回主logits
        logits = model(raw, feat)
        if isinstance(logits, tuple):
            logits = logits[0]
            
        loss = nn.CrossEntropyLoss()(logits, y)

        batch = y.size(0)
        total_loss += loss.item() * batch
        total_acc += accuracy(logits, y) * batch
        total_n += batch
        
        preds = logits.argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_targets.extend(y.cpu().numpy())
        
    return total_loss / total_n, total_acc / total_n, np.array(all_preds), np.array(all_targets)


def save_results(y_true: np.ndarray, y_pred: np.ndarray, model_name: str, 
                 dataset_type: str, output_dir: Path, le: LabelEncoder):
    """保存实验结果"""
    model_dir = output_dir / f'{model_name}_{dataset_type}'
    model_dir.mkdir(exist_ok=True)
    
    # 计算指标
    metrics = compute_metrics(y_true, y_pred)
    pd.DataFrame([metrics]).to_csv(model_dir / 'metrics.csv', index=False, encoding='utf-8-sig')
    
    # 混淆矩阵
    labels_full = list(range(len(le.classes_)))
    cm = confusion_matrix(y_true, y_pred, labels=labels_full)
    cm_df = pd.DataFrame(cm, 
                        index=[f'True_{i}' for i in range(cm.shape[0])], 
                        columns=[f'Pred_{i}' for i in range(cm.shape[1])])
    cm_df.to_csv(model_dir / 'confusion_matrix.csv', encoding='utf-8-sig')
    
    return metrics
