import os
from pathlib import Path
from typing import Tuple, Dict

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix


def load_datasets() -> Dict[str, Dict[str, pd.DataFrame]]:
    """加载三个数据集（复用 task2_1/datasets 下的数据）。"""
    base_dir = Path(__file__).resolve().parent
    src = base_dir / 'datasets'
    # 若 task2_2 下无 datasets，则回退到 task2_1/datasets
    if not src.exists():
        src = base_dir.parent / 'task2_1' / 'datasets'

    datasets = {}
    # Data1: 原始曲线
    data1_035 = pd.read_pickle(src / 'data1' / 'crimp_force_curves_dataset_035.pkl')
    data1_05 = pd.read_pickle(src / 'data1' / 'crimp_force_curves_dataset_05.pkl')
    datasets['data1'] = {'035': data1_035, '05': data1_05}

    # Data2: 特征提取
    data2_035 = pd.read_pickle(src / 'data2' / 'features_035.pkl')
    data2_05 = pd.read_pickle(src / 'data2' / 'features_05.pkl')
    datasets['data2'] = {'035': data2_035, '05': data2_05}

    # Data3: 特征筛选
    data3_035 = pd.read_pickle(src / 'data3' / 'features_035_selected.pkl')
    data3_05 = pd.read_pickle(src / 'data3' / 'features_05_selected.pkl')
    datasets['data3'] = {'035': data3_035, '05': data3_05}

    return datasets


def preprocess_data(df: pd.DataFrame, dataset_type: str) -> Tuple[np.ndarray, np.ndarray, StandardScaler]:
    """与 task2_1 对齐的预处理：
    - data1: 使用 'Force_curve_RoI' 作为 X
    - data2/3: 使用除标签列外的特征列作为 X
    - y 为 'Sub_label_encoded'
    - 返回标准化后的 X、原始 y、以及 scaler
    """
    if dataset_type == 'data1':
        X = np.array([row for row in df['Force_curve_RoI']])
        y = df['Sub_label_encoded'].values
    else:
        label_cols = ['CrimpID', 'Wire_cross-section_conductor', 'Main_label_string',
                      'Sub_label_string', 'Main-label_encoded', 'Sub_label_encoded',
                      'Binary_label_encoded', 'CFM_label_encoded']
        feat_cols = [c for c in df.columns if c not in label_cols]
        X = df[feat_cols].values
        y = df['Sub_label_encoded'].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y, scaler


def split_train_test(X: np.ndarray, y: np.ndarray, test_size: float = 0.2, seed: int = 42):
    return train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    report = classification_report(y_true, y_pred, output_dict=True)
    cm = confusion_matrix(y_true, y_pred)
    return {
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1_score': f1,
        'classification_report': report,
        'confusion_matrix': cm,
    }


def save_results_to_csv(results: Dict[str, float], out_csv: Path):
    row = {
        'accuracy': results['accuracy'],
        'precision': results['precision'],
        'recall': results['recall'],
        'f1_score': results['f1_score'],
    }
    df = pd.DataFrame([row])
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False, encoding='utf-8-sig')























