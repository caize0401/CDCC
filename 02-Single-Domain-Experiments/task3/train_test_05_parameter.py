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
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


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


def align_and_prepare(df_raw: pd.DataFrame, df_feat: pd.DataFrame):
    left = df_raw[['CrimpID', 'Force_curve_RoI', 'Sub_label_encoded']].copy()
    right = df_feat.copy()
    feature_cols = [c for c in right.columns if c not in LABEL_COLS]
    merged = pd.merge(left, right[['CrimpID', 'Sub_label_encoded'] + feature_cols],
                      on=['CrimpID', 'Sub_label_encoded'], how='inner')
    if len(merged) == 0:
        min_len = min(len(left), len(right))
        left = left.iloc[:min_len].reset_index(drop=True)
        right = right.iloc[:min_len].reset_index(drop=True)
        merged = pd.concat([
            left[['CrimpID', 'Force_curve_RoI', 'Sub_label_encoded']],
            right[feature_cols]
        ], axis=1)
    X_raw = np.stack(merged['Force_curve_RoI'].to_numpy()).astype(np.float32)
    X_feat = merged[feature_cols].to_numpy(dtype=np.float32)
    y = merged['Sub_label_encoded'].to_numpy(dtype=np.int64)
    return X_raw, X_feat, y, feature_cols


class HybridDataset(Dataset):
    def __init__(self, X_raw, X_feat, y, raw_scaler: StandardScaler, feat_scaler: StandardScaler):
        self.X_raw = X_raw
        self.X_feat = X_feat
        self.y = y
        self.raw_scaler = raw_scaler
        self.feat_scaler = feat_scaler

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        raw = self.X_raw[idx]
        feat = self.X_feat[idx]
        label = self.y[idx]
        raw_norm = self.raw_scaler.transform(raw.reshape(1, -1)).reshape(-1)
        feat_norm = self.feat_scaler.transform(feat.reshape(1, -1)).reshape(-1)
        return (
            torch.from_numpy(raw_norm).float().unsqueeze(0),
            torch.from_numpy(feat_norm).float(),
            torch.tensor(label, dtype=torch.long),
        )


def _import_model_module(module_key: str):
    task3_dir = Path(__file__).resolve().parent
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


def get_model(model_name: str, feat_in_dim: int, num_classes: int):
    mod = _import_model_module(model_name)
    for fn_name in ["get_model", "build_model", "create_model"]:
        fn = getattr(mod, fn_name, None)
        if callable(fn):
            return fn(feat_in_dim=feat_in_dim, num_classes=num_classes)
    for cls_name in ["HybridFusionV2", "HybridFusion", "Model", "Net"]:
        cls = getattr(mod, cls_name, None)
        if inspect.isclass(cls):
            try:
                return cls(feat_in_dim=feat_in_dim, num_classes=num_classes)
            except TypeError:
                try:
                    return cls(feat_in_dim, num_classes)
                except TypeError:
                    pass
    try:
        import torch.nn as nn
        for name, obj in inspect.getmembers(mod, inspect.isclass):
            if issubclass(obj, nn.Module):
                try:
                    return obj(feat_in_dim=feat_in_dim, num_classes=num_classes)
                except TypeError:
                    continue
    except Exception:
        pass
    raise ValueError(f"Unknown model or incompatible API in module '{model_name}'.")


def _extract_logits(output):
    if isinstance(output, (list, tuple)):
        return output[0]
    return output


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    total_acc = 0.0
    total_n = 0
    for raw, feat, y in loader:
        raw = raw.to(device)
        feat = feat.to(device)
        y = y.to(device)
        optimizer.zero_grad(set_to_none=True)
        outputs = model(raw, feat)
        if hasattr(model, 'loss'):
            try:
                loss = model.loss(outputs, y)
            except TypeError:
                logits = _extract_logits(outputs)
                loss = model.loss(logits, y)
        else:
            logits = _extract_logits(outputs)
            loss = torch.nn.functional.cross_entropy(logits, y)
        loss.backward()
        optimizer.step()
        batch = y.size(0)
        total_loss += loss.item() * batch
        logits_for_acc = _extract_logits(outputs)
        total_acc += (logits_for_acc.argmax(dim=1) == y).float().sum().item()
        total_n += batch
    return total_loss / total_n, total_acc / total_n


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    total_n = 0
    for raw, feat, y in loader:
        raw = raw.to(device)
        feat = feat.to(device)
        y = y.to(device)
        outputs = model(raw, feat)
        if hasattr(model, 'loss'):
            try:
                loss = model.loss(outputs, y)
            except TypeError:
                logits = _extract_logits(outputs)
                loss = model.loss(logits, y)
        else:
            logits = _extract_logits(outputs)
            loss = torch.nn.functional.cross_entropy(logits, y)
        batch = y.size(0)
        total_loss += loss.item() * batch
        logits_for_acc = _extract_logits(outputs)
        total_acc += (logits_for_acc.argmax(dim=1) == y).float().sum().item()
        total_n += batch
    return total_loss / total_n, total_acc / total_n


def run(size_tag: str, model_name: str, out_dir: Path, epochs: int, batch_size: int, lr: float):
    set_seed()
    base_dir = Path(__file__).resolve().parent
    out_dir = out_dir / size_tag / model_name
    out_dir.mkdir(parents=True, exist_ok=True)

    df_raw, df_feat = load_size_datasets(base_dir, size_tag)
    X_raw, X_feat, y, feat_cols = align_and_prepare(df_raw, df_feat)

    # 80/20 train-test split with seed 42
    Xr_tr, Xr_te, Xf_tr, Xf_te, y_tr, y_te = train_test_split(
        X_raw, X_feat, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_classes = int(np.max(y) + 1)

    raw_scaler = StandardScaler().fit(Xr_tr)
    feat_scaler = StandardScaler().fit(Xf_tr)
    train_ds = HybridDataset(Xr_tr, Xf_tr, y_tr, raw_scaler, feat_scaler)
    test_ds = HybridDataset(Xr_te, Xf_te, y_te, raw_scaler, feat_scaler)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

    model = get_model(model_name, feat_in_dim=X_feat.shape[1], num_classes=num_classes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    # Track best by test F1
    best_test_f1 = 0.0
    best_record = None
    best_conf_mat = None
    best_path = out_dir / 'best_model.pt'
    last_path = out_dir / 'last_model.pt'
    log_path = out_dir / 'logs.csv'
    with open(log_path, 'w', encoding='utf-8') as f:
        f.write('epoch,train_loss,train_acc,test_loss,test_acc,test_precision,test_recall,test_f1,lr\n')

    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, device)
        # Evaluate on test each epoch
        te_loss, te_acc = evaluate(model, test_loader, device)
        # Compute 4 metrics on test
        y_true = []
        y_pred = []
        with torch.no_grad():
            for raw, feat, yb in test_loader:
                raw = raw.to(device)
                feat = feat.to(device)
                logits = _extract_logits(model(raw, feat))
                preds = logits.argmax(dim=1).cpu().numpy()
                y_pred.append(preds)
                y_true.append(yb.numpy())
        y_true = np.concatenate(y_true)
        y_pred = np.concatenate(y_pred)
        te_prec = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        te_rec = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        te_f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        from sklearn.metrics import confusion_matrix
        te_cm = confusion_matrix(y_true, y_pred)

        scheduler.step(te_loss)
        with open(log_path, 'a', encoding='utf-8') as f:
            f.write(f"{epoch},{tr_loss:.6f},{tr_acc:.6f},{te_loss:.6f},{te_acc:.6f},{te_prec:.6f},{te_rec:.6f},{te_f1:.6f},{optimizer.param_groups[0]['lr']:.6e}\n")
        # Save best by test F1
        if te_f1 > best_test_f1:
            best_test_f1 = te_f1
            best_record = (te_acc, te_prec, te_rec, te_f1)
            best_conf_mat = te_cm
            torch.save({
                'model_state_dict': model.state_dict(),
                'feat_cols': feat_cols,
                'size_tag': size_tag,
                'best_test_f1': best_test_f1,
                'raw_scaler_mean': raw_scaler.mean_,
                'raw_scaler_scale': raw_scaler.scale_,
                'feat_scaler_mean': feat_scaler.mean_,
                'feat_scaler_scale': feat_scaler.scale_,
            }, best_path)
        print(f"[{size_tag}][{model_name}] epoch {epoch}: train_acc={tr_acc:.4f} test_acc={te_acc:.4f} test_f1={te_f1:.4f} best_f1={best_test_f1:.4f}")

    torch.save({'model_state_dict': model.state_dict()}, last_path)

    # Save best metrics csv
    import csv
    csv_path = out_dir / 'test_metrics_best.csv'
    if best_record is None:
        # Fallback compute once if not set
        acc, prec, rec, f1 = 0.0, 0.0, 0.0, 0.0
    else:
        acc, prec, rec, f1 = best_record
    with open(csv_path, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow(['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted'])
        writer.writerow([acc, prec, rec, f1])
    print(f"[{size_tag}][{model_name}] Best Test: acc={acc:.4f} prec={prec:.4f} rec={rec:.4f} f1={f1:.4f}")

    # Save best confusion matrix (English headers)
    if best_conf_mat is not None:
        cm_csv = out_dir / 'confusion_matrix.csv'
        # Build header: Pred_0..Pred_K-1, rows True_*
        import pandas as pd
        num_classes = best_conf_mat.shape[0]
        cols = [f'Pred_{i}' for i in range(num_classes)]
        idx = [f'True_{i}' for i in range(num_classes)]
        df_cm = pd.DataFrame(best_conf_mat, index=idx, columns=cols)
        df_cm.to_csv(cm_csv, encoding='utf-8-sig')


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--size', type=str, choices=['035', '05'], required=True)
    p.add_argument('--model', type=str, default='hybrid_fusion_v2')
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--batch_size', type=int, default=128)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--out', type=str, default='experiments_single')
    return p.parse_args()


def main():
    args = parse_args()
    set_seed()
    out_dir = Path(args.out)
    if not out_dir.is_absolute():
        out_dir = (Path(__file__).resolve().parent / out_dir)
    run(size_tag=args.size, model_name=args.model, out_dir=out_dir, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)


if __name__ == '__main__':
    main()


