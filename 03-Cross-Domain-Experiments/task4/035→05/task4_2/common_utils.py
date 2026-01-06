import os
from pathlib import Path
from typing import Tuple, Dict

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


LABEL_COLS = ['CrimpID', 'Wire_cross-section_conductor', 'Main_label_string',
              'Sub_label_string', 'Sub_label_encoded', 'Main-label_encoded',
              'Binary_label_encoded', 'CFM_label_encoded']


def load_datasets(base_dir: Path) -> Dict[str, Dict[str, pd.DataFrame]]:
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


def preprocess(df: pd.DataFrame, dataset_type: str) -> Tuple[np.ndarray, np.ndarray]:
    if dataset_type == 'data1':
        X = np.array([row for row in df['Force_curve_RoI']])
        y = df['Sub_label_encoded'].values
    else:
        feat_cols = [c for c in df.columns if c not in LABEL_COLS]
        X = df[feat_cols].values
        y = df['Sub_label_encoded'].values
    return X, y


def fit_scaler_on_union(X_train: np.ndarray, X_test: np.ndarray) -> StandardScaler:
    scaler = StandardScaler()
    scaler.fit(np.vstack([X_train, X_test]))
    return scaler


def transform_split(scaler: StandardScaler, X: np.ndarray) -> np.ndarray:
    return scaler.transform(X)


def encode_labels_union(y_train: np.ndarray, y_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray, LabelEncoder]:
    # 以“总数据集的5类”作为目标空间：用 train+test 做 fit，避免测试集中多出的类别映射不到
    le = LabelEncoder().fit(np.concatenate([y_train, y_test]))
    return le.transform(y_train), le.transform(y_test), le


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
        'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
        'f1_score': f1_score(y_true, y_pred, average='weighted', zero_division=0),
    }


