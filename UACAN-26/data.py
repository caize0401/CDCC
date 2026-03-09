"""
UACAN 双输入数据：data1 曲线 + data2 手工特征（35 维），按 CrimpID 对齐，支持域与类别筛选。
"""
from pathlib import Path
from typing import Tuple, List, Dict, Optional
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset
import torch

LABEL_COLS = [
    "CrimpID", "Wire_cross-section_conductor", "Main_label_string", "Sub_label_string",
    "Main-label_encoded", "Sub_label_encoded", "Binary_label_encoded", "CFM_label_encoded",
]

# data2 特征文件名：features_05.pkl / features_035.pkl
FEAT_FILE_NAMES = {"05": "features_05.pkl", "035": "features_035.pkl"}


def get_data_dirs(base_dir: Optional[Path] = None) -> Tuple[Path, Path]:
    """(data1 目录, data2 目录) — data2 为 35 维手工特征"""
    if base_dir is None:
        base_dir = Path(__file__).resolve().parent
    root = (base_dir / ".." / "..").resolve()
    return root / "datasets" / "data1", root / "datasets" / "data2"


def load_domain_dual(
    data1_dir: Path,
    data2_dir: Path,
    domain: str,
    selected_classes: List[int],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    加载指定域的曲线 + data2 手工特征（35 维），按 CrimpID 对齐，只保留 selected_classes。
    Returns:
        X_curve: (N, curve_len)
        X_feat: (N, 35)
        y_raw: (N,) 1-5
    """
    pkl1 = data1_dir / f"crimp_force_curves_dataset_{domain}.pkl"
    pkl2 = data2_dir / FEAT_FILE_NAMES.get(domain, f"features_{domain}.pkl")
    if not pkl2.exists():
        pkl2 = data2_dir / f"features_{domain}.pkl"
    if not pkl1.exists():
        raise FileNotFoundError(f"data1 不存在: {pkl1}")
    if not pkl2.exists():
        raise FileNotFoundError(f"data2 不存在: {pkl2}")
    df1 = pd.read_pickle(pkl1)
    df2 = pd.read_pickle(pkl2)
    # 标签用 data1 的 Sub_label_encoded
    label_col = "Sub_label_encoded" if "Sub_label_encoded" in df1.columns else "Sub_label"
    if label_col not in df1.columns:
        label_col = [c for c in df1.columns if "label" in c.lower() and "encoded" in c.lower()]
        label_col = label_col[0] if label_col else None
    if label_col is None:
        raise KeyError("data1 中未找到标签列")
    df1 = df1.copy()
    df1["_label"] = df1[label_col].astype(int)
    df1 = df1[df1["_label"].isin(selected_classes)].reset_index(drop=True)
    # 按 CrimpID 内连接
    merged = pd.merge(df1, df2, on="CrimpID", how="inner", suffixes=("_1", "_2"))
    y_raw = merged["_label"].values
    curves = merged["Force_curve_RoI"].values
    X_curve = np.stack([np.asarray(c).ravel() for c in curves]).astype(np.float32)
    # 特征列：取自 data2 的数值列（排除标签/ID），期望 35 维
    exclude = set(LABEL_COLS) | {"CrimpID", "Force_curve_RoI", "Force_curve_raw", "Force_curve_baseline"}
    feat_cols = [c for c in df2.columns if c not in exclude and c in merged.columns
                 and pd.api.types.is_numeric_dtype(merged[c])]
    if not feat_cols:
        feat_cols = [c for c in merged.columns if c not in ["CrimpID", "Force_curve_RoI", "_label"]
                     and pd.api.types.is_numeric_dtype(merged[c])]
    X_feat = merged[feat_cols].values.astype(np.float32)
    return X_curve, X_feat, y_raw


def build_label_mapping(
    source_classes: List[int],
    target_classes: List[int],
) -> Tuple[List[int], Dict[int, int], int, int]:
    common = sorted(set(source_classes) & set(target_classes))
    label_to_idx = {c: i for i, c in enumerate(common)}
    num_known = len(common)
    unknown_label = num_known
    return common, label_to_idx, unknown_label, num_known


def prepare_uacan_data(
    data1_dir: Path,
    data2_dir: Path,
    source_domain: str,
    target_domain: str,
    source_classes: List[int],
    target_classes: List[int],
) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray,
    np.ndarray, np.ndarray, np.ndarray, np.ndarray,
    List[int], Dict[int, int], int, int,
    StandardScaler, StandardScaler,
]:
    """
    返回:
        X_curve_s, X_feat_s, y_s, y_s_raw,
        X_curve_t, X_feat_t, y_t, y_t_raw,
        common, label_to_idx, num_known, unknown_label,
        scaler_curve, scaler_feat
    """
    common, label_to_idx, unknown_label, num_known = build_label_mapping(
        source_classes, target_classes
    )
    X_c_s, X_f_s, y_s_raw = load_domain_dual(data1_dir, data2_dir, source_domain, source_classes)
    X_c_t, X_f_t, y_t_raw = load_domain_dual(data1_dir, data2_dir, target_domain, target_classes)

    y_s = np.array([label_to_idx.get(int(l), unknown_label) for l in y_s_raw], dtype=np.int64)
    y_t = np.array([label_to_idx.get(int(l), unknown_label) for l in y_t_raw], dtype=np.int64)

    scaler_c = StandardScaler()
    scaler_f = StandardScaler()
    X_c_all = np.vstack([X_c_s, X_c_t])
    X_f_all = np.vstack([X_f_s, X_f_t])
    scaler_c.fit(X_c_all)
    scaler_f.fit(X_f_all)
    X_c_s = scaler_c.transform(X_c_s).astype(np.float32)
    X_c_t = scaler_c.transform(X_c_t).astype(np.float32)
    X_f_s = scaler_f.transform(X_f_s).astype(np.float32)
    X_f_t = scaler_f.transform(X_f_t).astype(np.float32)

    return (
        X_c_s, X_f_s, y_s, y_s_raw,
        X_c_t, X_f_t, y_t, y_t_raw,
        common, label_to_idx, num_known, unknown_label,
        scaler_c, scaler_f,
    )


def create_loaders(
    X_c_s: np.ndarray,
    X_f_s: np.ndarray,
    y_s: np.ndarray,
    X_c_t: np.ndarray,
    X_f_t: np.ndarray,
    y_t: np.ndarray,
    batch_size: int = 64,
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """训练 loader：源+目标，目标标签置 -1。测试 loader：仅目标。"""
    y_t_train = np.full(len(y_t), -1, dtype=np.int64)
    X_c = np.vstack([X_c_s, X_c_t])
    X_f = np.vstack([X_f_s, X_f_t])
    y_train = np.concatenate([y_s, y_t_train])
    d_train = np.concatenate([np.zeros(len(X_c_s), dtype=np.int64), np.ones(len(X_c_t), dtype=np.int64)])

    class _DS(Dataset):
        def __init__(self, Xc, Xf, y, d):
            self.Xc = torch.from_numpy(Xc).float()
            self.Xf = torch.from_numpy(Xf).float()
            self.y = torch.from_numpy(y).long()
            self.d = torch.from_numpy(d).long()

        def __len__(self):
            return len(self.y)

        def __getitem__(self, i):
            return self.Xc[i], self.Xf[i], self.y[i], self.d[i]

    train_loader = torch.utils.data.DataLoader(
        _DS(X_c, X_f, y_train, d_train), batch_size=batch_size, shuffle=True, num_workers=0
    )
    test_loader = torch.utils.data.DataLoader(
        _DS(X_c_t, X_f_t, y_t, np.ones(len(y_t), dtype=np.int64)),
        batch_size=batch_size, shuffle=False, num_workers=0
    )
    return train_loader, test_loader
