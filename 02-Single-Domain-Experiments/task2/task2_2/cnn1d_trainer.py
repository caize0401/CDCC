from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import accuracy_score

from common_utils import evaluate_predictions


class CurveDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = X.astype(np.float32)
        self.y = y.astype(np.int64)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x = torch.from_numpy(self.X[idx]).unsqueeze(0)  # (1, L)
        y = torch.tensor(self.y[idx], dtype=torch.long)
        return x, y


class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c, k, p):
        super().__init__()
        self.conv = nn.Conv1d(in_c, out_c, k, padding=p)
        self.bn = nn.BatchNorm1d(out_c)
        self.act = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool1d(2)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        x = self.pool(x)
        return x


class CNN1D(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.b1 = ConvBlock(1, 32, 7, 3)
        self.b2 = ConvBlock(32, 64, 5, 2)
        self.b3 = ConvBlock(64, 128, 3, 1)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.b1(x)
        x = self.b2(x)
        x = self.b3(x)
        x = self.gap(x).squeeze(-1)
        x = self.fc(x)
        return x


def train_epoch(model, loader, opt, device):
    model.train()
    ce = nn.CrossEntropyLoss()
    total, correct, loss_sum = 0, 0, 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        opt.zero_grad(set_to_none=True)
        logits = model(x)
        loss = ce(logits, y)
        loss.backward()
        opt.step()
        loss_sum += loss.item() * y.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        total += y.size(0)
    return loss_sum / total, correct / total


@torch.no_grad()
def eval_epoch(model, loader, device):
    model.eval()
    ce = nn.CrossEntropyLoss()
    total, correct, loss_sum = 0, 0, 0.0
    preds, trues = [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = ce(logits, y)
        loss_sum += loss.item() * y.size(0)
        pred = logits.argmax(1)
        correct += (pred == y).sum().item()
        total += y.size(0)
        preds.append(pred.cpu().numpy())
        trues.append(y.cpu().numpy())
    preds = np.concatenate(preds)
    trues = np.concatenate(trues)
    return loss_sum / total, correct / total, preds, trues


def train_and_eval_cnn1d(X_train, y_train, X_test, y_test, epochs: int = 50, batch_size: int = 128, lr: float = 1e-3) -> Dict[str, float]:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_classes = int(np.max(y_train) + 1)
    model = CNN1D(num_classes).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    tr_loader = DataLoader(CurveDataset(X_train, y_train), batch_size=batch_size, shuffle=True, num_workers=0)
    te_loader = DataLoader(CurveDataset(X_test, y_test), batch_size=batch_size, shuffle=False, num_workers=0)

    best_acc = 0.0
    for _ in range(epochs):
        train_epoch(model, tr_loader, opt, device)
        _, te_acc, preds, trues = eval_epoch(model, te_loader, device)
        if te_acc > best_acc:
            best_acc = te_acc
    # Final metrics with best (note: here we report last-epoch performance for simplicity)
    _, _, preds, trues = eval_epoch(model, te_loader, device)
    return evaluate_predictions(trues, preds)























