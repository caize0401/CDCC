"""
Task4_4通用工具函数
支持v4 (Transformer) 和 v5 (HybridFusion Pro) 模型的跨域实验
"""
import os
from pathlib import Path
from typing import Tuple, Dict, List, Optional
import sys
import importlib
import inspect

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
    }
    return datasets


def prepare_transformer_data(df_raw: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """
    为Transformer v4准备数据（只需要原始曲线）
    """
    # 提取原始力曲线和标签
    X_raw = np.stack(df_raw['Force_curve_RoI'].to_numpy()).astype(np.float32)
    y_raw = df_raw['Sub_label_encoded'].to_numpy(dtype=np.int64)
    
    # 重新编码标签，确保从0开始连续
    unique_labels = np.unique(y_raw)
    label_mapping = {old_label: new_label for new_label, old_label in enumerate(unique_labels)}
    y = np.array([label_mapping[label] for label in y_raw], dtype=np.int64)
    
    print(f"Transformer数据 - 原始曲线形状: {X_raw.shape}")
    print(f"Transformer数据 - 标签形状: {y.shape}")
    print(f"Transformer数据 - 原始标签: {unique_labels}")
    print(f"Transformer数据 - 重编码标签: {np.unique(y)}")
    print(f"Transformer数据 - 类别数: {len(np.unique(y))}")
    
    return X_raw, y


def align_and_prepare_hybrid_data(
    df_raw: pd.DataFrame, df_feat: pd.DataFrame
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """
    为混合模型v5准备数据（需要原始曲线+手工特征）
    """
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

    print(f"混合模型数据 - 合并后数据形状: {merged.shape}")
    
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
    y_raw = merged['Sub_label_encoded'].to_numpy(dtype=np.int64)

    # 重新编码标签，确保从0开始连续
    unique_labels = np.unique(y_raw)
    label_mapping = {old_label: new_label for new_label, old_label in enumerate(unique_labels)}
    y = np.array([label_mapping[label] for label in y_raw], dtype=np.int64)

    # Ensure shapes
    if X_raw.ndim != 2:
        raise ValueError(f"Expected raw shape (N, L), got {X_raw.shape}")
    
    print(f"混合模型数据 - 原始曲线形状: {X_raw.shape}")
    print(f"混合模型数据 - 手工特征形状: {X_feat.shape}")
    print(f"混合模型数据 - 标签形状: {y.shape}")
    print(f"混合模型数据 - 原始标签: {unique_labels}")
    print(f"混合模型数据 - 重编码标签: {np.unique(y)}")
    print(f"混合模型数据 - 类别数: {len(np.unique(y))}")
    
    return X_raw.astype(np.float32), X_feat, y, available_feature_cols


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


class TransformerDataset(Dataset):
    """专门用于Transformer v4模型的数据集类"""
    def __init__(self, X_raw: np.ndarray, y: np.ndarray, scaler: StandardScaler = None):
        self.X_raw = X_raw
        self.y = y
        self.scaler = scaler
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        raw = self.X_raw[idx]
        label = self.y[idx]
        
        # 标准化
        if self.scaler is not None:
            raw_norm = self.scaler.transform(raw.reshape(1, -1)).reshape(-1)
        else:
            raw_norm = raw
        
        return (
            torch.from_numpy(raw_norm).float(),  # (seq_len,)
            torch.tensor(label, dtype=torch.long)
        )


class HybridDataset(Dataset):
    """专门用于混合模型v5的数据集类"""
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


def _import_model_module(module_key: str):
    """导入模型模块"""
    # 添加task3路径以便导入模型
    task3_dir = Path(__file__).resolve().parent.parent / 'task3'
    if str(task3_dir) not in sys.path:
        sys.path.insert(0, str(task3_dir))
    
    candidates = [f"models.{module_key}", f"task3.models.{module_key}"]
    last_err = None
    for name in candidates:
        try:
            return importlib.import_module(name)
        except ModuleNotFoundError as e:
            last_err = e
    raise last_err if last_err else ModuleNotFoundError(module_key)


def get_transformer_model(model_name: str, input_dim: int, num_classes: int):
    """获取Transformer模型实例"""
    mod = _import_model_module(model_name)
    
    # 尝试常见的工厂函数
    for fn_name in ["get_model", "build_model", "create_model"]:
        fn = getattr(mod, fn_name, None)
        if callable(fn):
            try:
                return fn(input_dim=input_dim, num_classes=num_classes)
            except TypeError:
                pass
    
    # 尝试常见的类名
    for cls_name in ["TransformerV4", "Transformer", "TransformerModel", "Model", "Net"]:
        cls = getattr(mod, cls_name, None)
        if inspect.isclass(cls):
            try:
                return cls(input_dim=input_dim, num_classes=num_classes)
            except TypeError:
                try:
                    return cls(input_dim, num_classes)
                except TypeError:
                    pass
    
    # 尝试所有继承自nn.Module的类
    try:
        for name, obj in inspect.getmembers(mod, inspect.isclass):
            if issubclass(obj, nn.Module) and obj != nn.Module:
                try:
                    return obj(input_dim=input_dim, num_classes=num_classes)
                except TypeError:
                    continue
    except Exception:
        pass
    
    raise ValueError(f"无法在模块 '{model_name}' 中找到兼容的Transformer模型类。")


def get_hybrid_model(model_name: str, feat_in_dim: int, num_classes: int):
    """获取混合模型实例"""
    mod = _import_model_module(model_name)
    
    # 尝试常见的工厂函数
    for fn_name in ["get_model", "build_model", "create_model"]:
        fn = getattr(mod, fn_name, None)
        if callable(fn):
            try:
                return fn(feat_in_dim=feat_in_dim, num_classes=num_classes)
            except TypeError:
                pass
    
    # 尝试常见的类名
    for cls_name in ["HybridFusionPro", "HybridFusionV5", "HybridFusion", "Model", "Net"]:
        cls = getattr(mod, cls_name, None)
        if inspect.isclass(cls):
            try:
                return cls(feat_in_dim=feat_in_dim, num_classes=num_classes)
            except TypeError:
                try:
                    return cls(feat_in_dim, num_classes)
                except TypeError:
                    pass
    
    # 尝试所有继承自nn.Module的类
    try:
        for name, obj in inspect.getmembers(mod, inspect.isclass):
            if issubclass(obj, nn.Module) and obj != nn.Module:
                try:
                    return obj(feat_in_dim=feat_in_dim, num_classes=num_classes)
                except TypeError:
                    continue
    except Exception:
        pass
    
    raise ValueError(f"无法在模块 '{model_name}' 中找到兼容的混合模型类。")


def _extract_logits(output):
    """提取logits，处理可能的tuple输出"""
    if isinstance(output, (list, tuple)):
        return output[0]
    return output


def accuracy(pred: torch.Tensor, target: torch.Tensor) -> float:
    """计算准确率"""
    preds = pred.argmax(dim=1)
    return (preds == target).float().mean().item()


def save_results(y_true: np.ndarray, y_pred: np.ndarray, model_name: str, 
                 dataset_type: str, output_dir: Path, le: Optional[LabelEncoder] = None):
    """保存实验结果"""
    model_dir = output_dir / f'{model_name}_{dataset_type}'
    model_dir.mkdir(exist_ok=True)
    
    # 计算指标
    metrics = compute_metrics(y_true, y_pred)
    pd.DataFrame([metrics]).to_csv(model_dir / 'metrics.csv', index=False, encoding='utf-8-sig')
    
    # 混淆矩阵
    if le is not None:
        labels_full = list(range(len(le.classes_)))
    else:
        labels_full = list(range(len(np.unique(np.concatenate([y_true, y_pred])))))
    
    cm = confusion_matrix(y_true, y_pred, labels=labels_full)
    cm_df = pd.DataFrame(cm, 
                        index=[f'True_{i}' for i in range(cm.shape[0])], 
                        columns=[f'Pred_{i}' for i in range(cm.shape[1])])
    cm_df.to_csv(model_dir / 'confusion_matrix.csv', encoding='utf-8-sig')
    
    return metrics
