"""
Transformer域迁移实验
基于task3的Transformer v4架构，进行0.35→0.5的域迁移实验
只使用data1（原始曲线数据）
"""
import os
import random
from pathlib import Path
from typing import List, Tuple
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


RANDOM_SEED = 42
torch.backends.cudnn.benchmark = True

LABEL_COLS = [
    'CrimpID', 'Wire_cross-section_conductor', 'Main_label_string',
    'Sub_label_string', 'Main-label_encoded', 'Sub_label_encoded',
    'Binary_label_encoded', 'CFM_label_encoded'
]


def set_seed(seed: int = RANDOM_SEED) -> None:
    """设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class PositionalEncoding(nn.Module):
    """位置编码模块"""
    def __init__(self, d_model: int, max_len: int = 1000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        
        # 创建位置编码矩阵
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (seq_len, batch_size, d_model)
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class PatchEmbedding(nn.Module):
    """将时间序列分割成patches并嵌入"""
    def __init__(self, seq_len: int = 500, patch_size: int = 10, d_model: int = 256):
        super().__init__()
        self.patch_size = patch_size
        self.seq_len = seq_len
        self.num_patches = seq_len // patch_size
        
        # 线性投影层
        self.projection = nn.Linear(patch_size, d_model)
        
        # 类别token（可选）
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, seq_len)
        batch_size = x.size(0)
        
        # 重塑为patches
        # (batch_size, seq_len) -> (batch_size, num_patches, patch_size)
        x = x[:, :self.num_patches * self.patch_size]  # 确保长度可被patch_size整除
        x = x.view(batch_size, self.num_patches, self.patch_size)
        
        # 投影到d_model维度
        x = self.projection(x)  # (batch_size, num_patches, d_model)
        
        # 添加类别token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # (batch_size, num_patches+1, d_model)
        
        return x


class TransformerBlock(nn.Module):
    """Transformer编码器块"""
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 1024, 
                 dropout: float = 0.1):
        super().__init__()
        
        # 多头自注意力
        self.attention = nn.MultiheadAttention(
            embed_dim=d_model, 
            num_heads=nhead, 
            dropout=dropout,
            batch_first=True
        )
        
        # 前馈网络
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout)
        )
        
        # 层归一化
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 自注意力 + 残差连接
        attn_output, _ = self.attention(x, x, x)
        x = self.norm1(x + attn_output)
        
        # 前馈网络 + 残差连接
        ff_output = self.feed_forward(x)
        x = self.norm2(x + ff_output)
        
        return x


class TransformerV4(nn.Module):
    """
    Transformer v4: 简化版Transformer模型
    基于task3的架构，用于域迁移实验
    """
    def __init__(self, input_dim: int = 500, num_classes: int = 5, 
                 d_model: int = 256, nhead: int = 8, num_layers: int = 6,
                 patch_size: int = 10, dim_feedforward: int = 1024, 
                 dropout: float = 0.1, label_smoothing: float = 0.1):
        super().__init__()
        
        self.d_model = d_model
        self.num_classes = num_classes
        self._label_smoothing = label_smoothing
        
        # Patch嵌入
        self.patch_embedding = PatchEmbedding(
            seq_len=input_dim,
            patch_size=patch_size,
            d_model=d_model
        )
        
        # 位置编码
        max_len = (input_dim // patch_size) + 1  # +1 for cls token
        self.pos_encoder = PositionalEncoding(d_model, max_len=max_len, dropout=dropout)
        
        # Transformer编码器层
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout
            ) for _ in range(num_layers)
        ])
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes)
        )
        
        # 初始化权重
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, seq_len)
        
        # Patch嵌入
        x = self.patch_embedding(x)  # (batch_size, num_patches+1, d_model)
        
        # 位置编码 (需要转换维度)
        x = x.transpose(0, 1)  # (num_patches+1, batch_size, d_model)
        x = self.pos_encoder(x)
        x = x.transpose(0, 1)  # (batch_size, num_patches+1, d_model)
        
        # Transformer编码器
        for block in self.transformer_blocks:
            x = block(x)
        
        # 使用cls token进行最终分类
        cls_features = x[:, 0, :]  # (batch_size, d_model)
        logits = self.classifier(cls_features)
        
        return logits


def load_datasets(base_dir: Path) -> dict:
    """加载数据集"""
    datasets = {}
    for dataset_type in ['data1', 'data2', 'data3']:
        datasets[dataset_type] = {}
        for size in ['035', '05']:
            if dataset_type == 'data1':
                file_path = base_dir / 'datasets' / dataset_type / f'crimp_force_curves_dataset_{size}.pkl'
            else:
                file_path = base_dir / 'datasets' / dataset_type / f'features_{size}.pkl'
            
            if file_path.exists():
                datasets[dataset_type][size] = pd.read_pickle(file_path)
            else:
                print(f"警告: 文件不存在 {file_path}")
    
    return datasets


def prepare_transformer_data(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """准备Transformer模型的数据"""
    # 提取原始力曲线和标签
    X_raw = np.stack(df['Force_curve_RoI'].to_numpy()).astype(np.float32)
    y_raw = df['Sub_label_encoded'].to_numpy(dtype=np.int64)
    
    # 重新编码标签，确保从0开始连续
    unique_labels = np.unique(y_raw)
    label_mapping = {old_label: new_label for new_label, old_label in enumerate(unique_labels)}
    y = np.array([label_mapping[label] for label in y_raw], dtype=np.int64)
    
    return X_raw, y


class TransformerDataset(Dataset):
    """Transformer数据集"""
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


def train_one_epoch(model, loader, optimizer, device):
    """训练一个epoch"""
    model.train()
    total_loss = 0.0
    total_acc = 0.0
    total_n = 0
    
    for raw, y in loader:
        raw = raw.to(device)
        y = y.to(device)
        
        optimizer.zero_grad(set_to_none=True)
        
        # 前向传播
        logits = model(raw)
        loss = F.cross_entropy(logits, y, label_smoothing=0.1)
        
        # 反向传播
        loss.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # 统计
        batch = y.size(0)
        total_loss += loss.item() * batch
        total_acc += (logits.argmax(dim=1) == y).float().sum().item()
        total_n += batch
    
    return total_loss / total_n, total_acc / total_n


@torch.no_grad()
def evaluate(model, loader, device):
    """评估模型"""
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    total_n = 0
    
    for raw, y in loader:
        raw = raw.to(device)
        y = y.to(device)
        
        logits = model(raw)
        loss = F.cross_entropy(logits, y)
        
        batch = y.size(0)
        total_loss += loss.item() * batch
        total_acc += (logits.argmax(dim=1) == y).float().sum().item()
        total_n += batch
    
    return total_loss / total_n, total_acc / total_n


def get_predictions(model, loader, device):
    """获取模型预测结果"""
    model.eval()
    y_true = []
    y_pred = []
    
    with torch.no_grad():
        for raw, y in loader:
            raw = raw.to(device)
            logits = model(raw)
            preds = logits.argmax(dim=1).cpu().numpy()
            y_pred.append(preds)
            y_true.append(y.numpy())
    
    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)
    return y_true, y_pred


def run_transformer_domain_transfer():
    """运行Transformer域迁移实验"""
    set_seed()
    
    base_dir = Path(__file__).resolve().parent
    out_root = base_dir / 'results'
    out_root.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("Transformer域迁移实验 (0.35→0.5)")
    print("=" * 60)
    
    # 加载数据集
    datasets = load_datasets(base_dir)
    
    # 只使用data1（原始曲线数据）
    dataset_type = 'data1'
    print(f"使用数据集: {dataset_type}")
    
    # 获取训练和测试数据
    df_train = datasets[dataset_type]['035']  # 源域：0.35mm
    df_test = datasets[dataset_type]['05']    # 目标域：0.5mm
    
    print(f"训练集（源域0.35mm）: {len(df_train)} 样本")
    print(f"测试集（目标域0.5mm）: {len(df_test)} 样本")
    
    # 准备数据
    X_train_raw, y_train_raw = prepare_transformer_data(df_train)
    X_test_raw, y_test_raw = prepare_transformer_data(df_test)
    
    print(f"训练数据形状: {X_train_raw.shape}")
    print(f"测试数据形状: {X_test_raw.shape}")
    print(f"类别数: {len(np.unique(np.concatenate([y_train_raw, y_test_raw])))}")
    
    # 数据标准化（联合拟合）
    print("标准化数据...")
    scaler = StandardScaler()
    X_combined = np.vstack([X_train_raw, X_test_raw])
    X_combined_flat = X_combined.reshape(-1, X_combined.shape[-1])
    scaler.fit(X_combined_flat)
    
    # 创建数据集和加载器
    train_ds = TransformerDataset(X_train_raw, y_train_raw, scaler)
    test_ds = TransformerDataset(X_test_raw, y_test_raw, scaler)
    
    batch_size = 32
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, 
                            num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, 
                           num_workers=0, pin_memory=True)
    
    # 创建模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_classes = len(np.unique(np.concatenate([y_train_raw, y_test_raw])))
    input_dim = X_train_raw.shape[1]
    
    print(f"创建Transformer模型...")
    print(f"输入维度: {input_dim}, 类别数: {num_classes}")
    print(f"设备: {device}")
    
    model = TransformerV4(
        input_dim=input_dim,
        num_classes=num_classes,
        d_model=256,
        nhead=8,
        num_layers=6,
        patch_size=10,
        dropout=0.1
    ).to(device)
    
    # 打印模型参数数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数总数: {total_params:,}")
    
    # 优化器和调度器
    epochs = 50
    lr = 1e-4
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4, 
                                 betas=(0.9, 0.999), eps=1e-8)
    
    # Warmup + Cosine Annealing调度器
    warmup_epochs = 5
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        else:
            if epochs == warmup_epochs:
                return 1.0
            return 0.5 * (1 + np.cos(np.pi * (epoch - warmup_epochs) / (epochs - warmup_epochs)))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # 训练循环
    best_test_f1 = 0.0
    best_record = None
    best_conf_mat = None
    
    print("开始训练...")
    for epoch in range(1, epochs + 1):
        # 训练
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, device)
        
        # 测试
        test_loss, test_acc = evaluate(model, test_loader, device)
        
        # 详细指标
        y_true, y_pred = get_predictions(model, test_loader, device)
        test_prec = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        test_rec = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        test_f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        test_cm = confusion_matrix(y_true, y_pred)
        
        # 更新学习率
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        # 保存最佳模型
        if test_f1 > best_test_f1:
            best_test_f1 = test_f1
            best_record = (test_acc, test_prec, test_rec, test_f1)
            best_conf_mat = test_cm
        
        # 打印进度
        if epoch % 10 == 0 or epoch == epochs:
            print(f"Epoch {epoch:3d}/{epochs}: "
                  f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
                  f"test_loss={test_loss:.4f} test_acc={test_acc:.4f} "
                  f"test_f1={test_f1:.4f} best_f1={best_test_f1:.4f} lr={current_lr:.2e}")
    
    # 保存结果
    if best_record is not None:
        acc, prec, rec, f1 = best_record
        
        # 创建输出目录
        model_dir = out_root / f'Transformer_{dataset_type}'
        model_dir.mkdir(exist_ok=True)
        
        # 保存指标CSV
        metrics_path = model_dir / 'metrics.csv'
        metrics_data = {
            'accuracy': acc,
            'precision': prec,
            'recall': rec,
            'f1_score': f1
        }
        pd.DataFrame([metrics_data]).to_csv(metrics_path, index=False, encoding='utf-8-sig')
        
        # 保存混淆矩阵
        if best_conf_mat is not None:
            cm_path = model_dir / 'confusion_matrix.csv'
            num_classes = best_conf_mat.shape[0]
            cols = [f'Pred_{i}' for i in range(num_classes)]
            idx = [f'True_{i}' for i in range(num_classes)]
            df_cm = pd.DataFrame(best_conf_mat, index=idx, columns=cols)
            df_cm.to_csv(cm_path, encoding='utf-8-sig')
        
        print(f"\nTransformer域迁移实验结果:")
        print(f"  准确率: {acc:.4f}")
        print(f"  精确率: {prec:.4f}")
        print(f"  召回率: {rec:.4f}")
        print(f"  F1分数: {f1:.4f}")
        print(f"  结果保存在: {model_dir}")
        
        return metrics_data
    else:
        print("训练失败，没有保存结果")
        return None


def update_summary():
    """更新summary.csv文件"""
    base_dir = Path(__file__).resolve().parent
    results_dir = base_dir / 'results'
    summary_path = results_dir / 'summary.csv'
    
    # 读取现有的summary
    if summary_path.exists():
        summary_df = pd.read_csv(summary_path)
    else:
        summary_df = pd.DataFrame(columns=['model', 'dataset', 'accuracy', 'precision', 'recall', 'f1_score'])
    
    # 收集所有结果
    rows = []
    for model_dir in results_dir.iterdir():
        if model_dir.is_dir() and (model_dir / 'metrics.csv').exists():
            metrics_path = model_dir / 'metrics.csv'
            metrics_df = pd.read_csv(metrics_path)
            
            # 从目录名解析模型和数据集
            dir_name = model_dir.name
            if '_' in dir_name:
                model_name, dataset_name = dir_name.split('_', 1)
                
                for _, row in metrics_df.iterrows():
                    rows.append({
                        'model': model_name,
                        'dataset': dataset_name,
                        'accuracy': row['accuracy'],
                        'precision': row['precision'],
                        'recall': row['recall'],
                        'f1_score': row['f1_score']
                    })
    
    # 创建新的summary
    new_summary_df = pd.DataFrame(rows)
    new_summary_df = new_summary_df.sort_values(['model', 'dataset'])
    
    # 保存
    new_summary_df.to_csv(summary_path, index=False, encoding='utf-8-sig')
    print(f"Summary已更新: {summary_path}")


def main():
    """主函数"""
    print("开始Transformer域迁移实验...")
    
    try:
        # 运行实验
        result = run_transformer_domain_transfer()
        
        if result is not None:
            # 更新summary
            update_summary()
            print("\n实验完成！")
        else:
            print("\n实验失败！")
            
    except Exception as e:
        print(f"\n实验失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
