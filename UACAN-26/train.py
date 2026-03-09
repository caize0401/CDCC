"""
UACAN 训练：双输入(曲线+特征)，条件域对抗 + cwMMD + 能量未知检测 + 动态调度。
每 epoch 输出源域 ACC、目标域 ACC、H-SCORE；训练结束保存目标域混淆矩阵(Excel)。
"""
import argparse
import random
from pathlib import Path
from typing import List, Tuple
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from config import (
    SOURCE_DOMAIN, TARGET_DOMAIN, SOURCE_CLASSES, TARGET_CLASSES,
    LAMBDA_0, ENERGY_TEMP, ENERGY_THRESHOLD, ENERGY_MARGIN, LAMBDA_UNK,
)
from data import get_data_dirs, prepare_uacan_data, create_loaders
from model import UACAN


RANDOM_SEED = 42


def set_seed(seed: int = RANDOM_SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_class_list(s: str) -> List[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def get_alpha_schedule(epoch: int, max_epochs: int, gamma: float = 10.0) -> float:
    p = epoch / max(1, max_epochs)
    return 2.0 / (1.0 + np.exp(-gamma * p)) - 1.0


def accuracy(logits: torch.Tensor, targets: torch.Tensor, ignore_index: int = -1) -> float:
    if targets.numel() == 0:
        return 0.0
    mask = targets >= 0 if ignore_index == -1 else (targets != ignore_index)
    if not mask.any():
        return 0.0
    pred = logits.argmax(dim=1)
    return (pred[mask] == targets[mask]).float().mean().item()


def h_score(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_known: int,
    unknown_label: int,
) -> float:
    known_mask = (y_true >= 0) & (y_true < num_known)
    unknown_mask = y_true == unknown_label
    known_acc = (y_pred[known_mask] == y_true[known_mask]).mean() if known_mask.sum() > 0 else 0.0
    unknown_acc = (y_pred[unknown_mask] == unknown_label).mean() if unknown_mask.sum() > 0 else 1.0
    if known_acc + unknown_acc == 0:
        return 0.0
    return 2.0 * known_acc * unknown_acc / (known_acc + unknown_acc)


def train_epoch(
    model: UACAN,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    max_epochs: int,
    num_known: int,
    lambda_0: float,
    T: float,
    delta: float,
    margin: float,
    lambda_unk: float,
) -> Tuple[float, float]:
    model.train()
    total_loss = 0.0
    total_acc = 0.0
    total_n = 0
    labeled_n = 0
    alpha = get_alpha_schedule(epoch, max_epochs)

    for x_c, x_f, y, d in loader:
        x_c, x_f, y, d = x_c.to(device), x_f.to(device), y.to(device), d.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits, domain_logits, F = model(x_c, x_f, alpha=alpha)

        mask_s = d == 0
        mask_t = d == 1
        has_s, has_t = mask_s.any().item(), mask_t.any().item()

        logits_s = logits[mask_s]
        logits_t = logits[mask_t]
        domain_s = domain_logits[mask_s]
        domain_t = domain_logits[mask_t]
        F_s = F[mask_s]
        F_t = F[mask_t]
        y_s = y[mask_s]

        if not has_s and not has_t:
            continue

        # 动态调度
        if has_t:
            P_t = torch.softmax(logits_t, dim=1)
            w_k = P_t.mean(dim=0)
            var_w = w_k.var().item() if logits_t.size(0) > 0 else 0.0
            E_t = model.energy(logits_t, T)
            rho = (E_t > delta).float().mean().item()
        else:
            var_w, rho = 0.0, 0.0
        lambda_cwmmd = lambda_0 * (1.0 - rho)
        lambda_domain = lambda_0 * (1.0 - min(var_w, 1.0))

        if has_s and has_t and logits_s.size(0) > 0 and logits_t.size(0) > 0:
            loss, _ = model.compute_loss(
                logits_s, domain_s, logits_t, domain_t,
                F_s, F_t, y_s, d[mask_s], d[mask_t],
                lambda_domain=lambda_domain,
                lambda_cwmmd=lambda_cwmmd,
                lambda_unk=lambda_unk,
                T=T, delta=delta, margin=margin,
            )
        elif has_s and logits_s.size(0) > 0:
            labeled = y_s >= 0
            if labeled.any():
                loss = torch.nn.functional.cross_entropy(
                    logits_s[labeled], y_s[labeled].clamp(0, num_known - 1)
                )
            else:
                loss = torch.tensor(0.0, device=device)
        elif has_t and logits_t.size(0) > 0:
            E_t = model.energy(logits_t, T)
            high_e = E_t > delta
            if high_e.any():
                loss = lambda_unk * torch.nn.functional.relu(margin - E_t[high_e]).mean()
            else:
                loss = torch.tensor(0.0, device=device)
        else:
            loss = torch.tensor(0.0, device=device)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        n = x_c.size(0)
        total_loss += loss.item() * n
        acc = accuracy(logits, y, ignore_index=-1)
        labeled_n += (y >= 0).sum().item()
        total_acc += acc * (y >= 0).sum().item()
        total_n += n

    avg_loss = total_loss / total_n if total_n else 0.0
    avg_acc = total_acc / max(labeled_n, 1)
    return avg_loss, avg_acc


@torch.no_grad()
def evaluate(
    model: UACAN,
    loader: DataLoader,
    device: torch.device,
    num_known: int,
    unknown_label: int,
    delta: float,
    T: float = 1.0,
) -> Tuple[float, float, np.ndarray, np.ndarray]:
    """预测：E(x)>delta 判为 unknown_label，否则 argmax。返回 ACC、H-SCORE、y_true、y_pred。"""
    model.eval()
    all_pred, all_true = [], []
    for x_c, x_f, y, _ in loader:
        x_c, x_f = x_c.to(device), x_f.to(device)
        logits, _, _ = model(x_c, x_f, alpha=0.0)
        E = model.energy(logits, T).cpu().numpy()
        pred = logits.argmax(dim=1).cpu().numpy()
        pred[E > delta] = unknown_label
        all_pred.append(pred)
        all_true.append(y.numpy())
    y_true = np.concatenate(all_true)
    y_pred = np.concatenate(all_pred)
    mask = y_true >= 0
    acc = (y_pred[mask] == y_true[mask]).mean() if mask.any() else 0.0
    h = h_score(y_true, y_pred, num_known, unknown_label)
    return acc, h, y_true, y_pred


def main():
    parser = argparse.ArgumentParser(description="UACAN 双输入域自适应")
    parser.add_argument("--source_domain", type=str, default=SOURCE_DOMAIN, choices=["05", "035"])
    parser.add_argument("--target_domain", type=str, default=TARGET_DOMAIN, choices=["05", "035"])
    parser.add_argument("--source_classes", type=str, default=",".join(map(str, SOURCE_CLASSES)))
    parser.add_argument("--target_classes", type=str, default=",".join(map(str, TARGET_CLASSES)))
    parser.add_argument("--data_dir", type=str, default=None, help="父目录 2026/datasets，含 data1/data2")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lambda_0", type=float, default=LAMBDA_0)
    parser.add_argument("--energy_T", type=float, default=ENERGY_TEMP)
    parser.add_argument("--energy_delta", type=float, default=ENERGY_THRESHOLD)
    parser.add_argument("--energy_margin", type=float, default=ENERGY_MARGIN)
    parser.add_argument("--lambda_unk", type=float, default=LAMBDA_UNK)
    parser.add_argument("--out_dir", type=str, default="experiments")
    args = parser.parse_args()

    source_domain = args.source_domain
    target_domain = args.target_domain
    source_classes = parse_class_list(args.source_classes)
    target_classes = parse_class_list(args.target_classes)
    if source_domain == target_domain:
        raise SystemExit("源域与目标域不能相同")

    base_dir = Path(__file__).resolve().parent
    if args.data_dir:
        root = Path(args.data_dir)
        data1_dir, data2_dir = root / "data1", root / "data2"
    else:
        data1_dir, data2_dir = get_data_dirs(base_dir)
    out_dir = base_dir / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    set_seed()

    print("加载数据（data1 曲线 + data2 手工特征 35 维）...")
    (
        X_c_s, X_f_s, y_s, _,
        X_c_t, X_f_t, y_t, _,
        common, label_to_idx, num_known, unknown_label,
        scaler_c, scaler_f,
    ) = prepare_uacan_data(
        data1_dir, data2_dir, source_domain, target_domain,
        source_classes, target_classes,
    )
    train_loader, test_loader = create_loaders(
        X_c_s, X_f_s, y_s, X_c_t, X_f_t, y_t, batch_size=args.batch_size
    )

    curve_dim = X_c_s.shape[1]
    feat_dim = X_f_s.shape[1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UACAN(
        curve_dim=curve_dim,
        feat_dim=feat_dim,
        num_classes=num_known,
        hidden_dim=128,
        fused_dim=256,
        num_domains=2,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    src_loader = DataLoader(
        torch.utils.data.TensorDataset(
            torch.from_numpy(X_c_s).float(),
            torch.from_numpy(X_f_s).float(),
            torch.from_numpy(y_s).long(),
            torch.zeros(len(X_c_s), dtype=torch.long),
        ),
        batch_size=args.batch_size, shuffle=False
    )
    tgt_loader = DataLoader(
        torch.utils.data.TensorDataset(
            torch.from_numpy(X_c_t).float(),
            torch.from_numpy(X_f_t).float(),
            torch.from_numpy(y_t).long(),
            torch.ones(len(X_c_t), dtype=torch.long),
        ),
        batch_size=args.batch_size, shuffle=False
    )

    log_path = out_dir / "training_log.csv"
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("epoch,source_acc,target_acc,h_score,train_loss,train_source_acc\n")

    print(f"源域: {source_domain}, 目标域: {target_domain}")
    print(f"源域类别: {source_classes}, 目标域类别: {target_classes}")
    print(f"公共类别: {common}, K={num_known}, 未知类索引={unknown_label}")
    print(f"曲线维度: {curve_dim}, 特征维度: {feat_dim}")
    print("每 epoch: Source ACC, Target ACC, H-SCORE\n")

    for epoch in range(1, args.epochs + 1):
        train_loss, train_src_acc = train_epoch(
            model, train_loader, optimizer, device, epoch, args.epochs, num_known,
            lambda_0=args.lambda_0, T=args.energy_T, delta=args.energy_delta,
            margin=args.energy_margin, lambda_unk=args.lambda_unk,
        )
        src_acc, _, _, _ = evaluate(
            model, src_loader, device, num_known, unknown_label, args.energy_delta, args.energy_T
        )
        tgt_acc, h, y_t_true, y_t_pred = evaluate(
            model, tgt_loader, device, num_known, unknown_label, args.energy_delta, args.energy_T
        )
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"{epoch},{src_acc:.6f},{tgt_acc:.6f},{h:.6f},{train_loss:.6f},{train_src_acc:.6f}\n")
        print(
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"Source ACC: {src_acc:.4f} | Target ACC: {tgt_acc:.4f} | H-SCORE: {h:.4f} | "
            f"Train Loss: {train_loss:.4f}"
        )

    ckpt_path = out_dir / "uacan_best.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "num_known": num_known,
        "unknown_label": unknown_label,
        "common_classes": common,
        "curve_dim": curve_dim,
        "feat_dim": feat_dim,
    }, ckpt_path)

    # 目标域混淆矩阵 -> Excel
    from sklearn.metrics import confusion_matrix
    class_names = [f"class_{common[i]}" for i in range(num_known)] + ["unknown"]
    labels_cm = list(range(num_known + 1))
    cm = confusion_matrix(y_t_true, y_t_pred, labels=labels_cm)
    cm_df = pd.DataFrame(
        cm,
        index=[f"True_{class_names[i]}" for i in range(len(class_names))],
        columns=[f"Pred_{class_names[j]}" for j in range(len(class_names))],
    )
    excel_path = out_dir / "target_confusion_matrix.xlsx"
    try:
        cm_df.to_excel(excel_path, sheet_name="confusion_matrix")
        print(f"\n目标域混淆矩阵已保存: {excel_path}")
    except Exception as e:
        csv_path = out_dir / "target_confusion_matrix.csv"
        cm_df.to_csv(csv_path, encoding="utf-8-sig")
        print(f"\n目标域混淆矩阵(CSV): {csv_path} ({e})")

    print("\n训练完成。运行说明见 README。")


if __name__ == "__main__":
    main()
