from typing import Dict

import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# Classical models
def make_mlp() -> MLPClassifier:
    return MLPClassifier(hidden_layer_sizes=(256, 128), activation='relu', solver='adam',
                         alpha=1e-4, max_iter=500, random_state=42)


def make_rf() -> RandomForestClassifier:
    return RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)


def make_xgb() -> XGBClassifier:
    return XGBClassifier(
        n_estimators=400, max_depth=6, learning_rate=0.05, subsample=0.9, colsample_bytree=0.9,
        objective='multi:softprob', eval_metric='mlogloss', tree_method='hist', random_state=42, n_jobs=-1
    )


def make_automl_voter() -> VotingClassifier:
    rf = make_rf()
    gb = None
    try:
        from sklearn.ensemble import GradientBoostingClassifier
        gb = GradientBoostingClassifier(n_estimators=200, learning_rate=0.05, random_state=42)
    except Exception:
        pass
    lr = LogisticRegression(max_iter=1000, random_state=42)
    svc = SVC(probability=True, random_state=42)
    knn = KNeighborsClassifier(n_neighbors=7)
    est = [("rf", rf), ("lr", lr), ("svc", svc), ("knn", knn)]
    if gb is not None:
        est.append(("gb", gb))
    return VotingClassifier(estimators=est, voting='soft', n_jobs=-1)


# CNN for data1 (raw ROI)
class CurveDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = X.astype(np.float32)
        self.y = y.astype(np.int64)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        import torch
        x = torch.from_numpy(self.X[idx]).unsqueeze(0)
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


def train_eval_cnn1d(X_train, y_train, X_test, y_test, epochs=50, batch_size=128, lr=1e-3) -> dict:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_classes = int(np.max(y_train) + 1)
    model = CNN1D(num_classes).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    ce = nn.CrossEntropyLoss()

    tr_loader = DataLoader(CurveDataset(X_train, y_train), batch_size=batch_size, shuffle=True, num_workers=0)
    te_loader = DataLoader(CurveDataset(X_test, y_test), batch_size=batch_size, shuffle=False, num_workers=0)

    best = 0.0
    for _ in range(epochs):
        model.train()
        for x, y in tr_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(x)
            loss = ce(logits, y)
            loss.backward()
            opt.step()

        # quick eval
        model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for x, y in te_loader:
                x = x.to(device)
                logits = model(x)
                preds.append(logits.argmax(1).cpu().numpy())
                trues.append(y.numpy())
        preds = np.concatenate(preds)
        trues = np.concatenate(trues)
        acc = (preds == trues).mean()
        best = max(best, acc)

    from sklearn.metrics import precision_score, recall_score, f1_score
    prec = precision_score(trues, preds, average='weighted', zero_division=0)
    rec = recall_score(trues, preds, average='weighted', zero_division=0)
    f1 = f1_score(trues, preds, average='weighted', zero_division=0)
    return {'accuracy': acc, 'precision': prec, 'recall': rec, 'f1_score': f1, 'y_true': trues, 'y_pred': preds}


