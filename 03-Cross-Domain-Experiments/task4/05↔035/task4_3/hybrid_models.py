"""
混合模型定义 - 从task3复制并适配跨域实验
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ==================== HybridFusion (v1) ====================

class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, pool: int, dropout: float = 0.1):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding)
        self.bn = nn.BatchNorm1d(out_channels)
        self.act = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool1d(pool)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        x = self.pool(x)
        x = self.drop(x)
        return x


class SEBlock(nn.Module):
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        self.fc1 = nn.Linear(channels, channels // reduction)
        self.fc2 = nn.Linear(channels // reduction, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, l = x.shape
        s = x.mean(dim=2)
        s = F.relu(self.fc1(s), inplace=True)
        s = torch.sigmoid(self.fc2(s))
        s = s.view(b, c, 1)
        return x * s


class CNNBackbone(nn.Module):
    def __init__(self, in_channels: int = 1, base_channels: int = 48, out_dim: int = 160, dropout: float = 0.15):
        super().__init__()
        self.block1 = ConvBlock(in_channels, base_channels, kernel_size=7, pool=2, dropout=dropout)
        self.block2 = ConvBlock(base_channels, base_channels * 2, kernel_size=5, pool=2, dropout=dropout)
        self.se2 = SEBlock(base_channels * 2)
        self.block3 = ConvBlock(base_channels * 2, base_channels * 4, kernel_size=3, pool=2, dropout=dropout)
        self.se3 = SEBlock(base_channels * 4)
        self.block4 = ConvBlock(base_channels * 4, base_channels * 4, kernel_size=3, pool=2, dropout=dropout)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.proj = nn.Linear(base_channels * 4, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.block2(x)
        x = self.se2(x)
        x = self.block3(x)
        x = self.se3(x)
        x = self.block4(x)
        x = self.gap(x).squeeze(-1)
        x = F.relu(self.proj(x), inplace=True)
        return x


class FeatureMLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 160, out_dim: int = 80, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class HybridFusion(nn.Module):
    def __init__(self, feat_in_dim: int, num_classes: int = 5, cnn_out: int = 160, feat_out: int = 80, fusion_hidden: int = 160, label_smoothing: float = 0.0):
        super().__init__()
        self.cnn = CNNBackbone(in_channels=1, base_channels=48, out_dim=cnn_out, dropout=0.15)
        self.mlp = FeatureMLP(in_dim=feat_in_dim, hidden=160, out_dim=feat_out, dropout=0.2)
        self.fusion = nn.Sequential(
            nn.Linear(cnn_out + feat_out, fusion_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
        )
        self.classifier = nn.Linear(fusion_hidden, num_classes)
        self._label_smoothing = label_smoothing

    def forward(self, raw_1d: torch.Tensor, feat_vec: torch.Tensor) -> torch.Tensor:
        f_cnn = self.cnn(raw_1d)
        f_feat = self.mlp(feat_vec)
        fused = torch.cat([f_cnn, f_feat], dim=1)
        shared = self.fusion(fused)
        logits = self.classifier(shared)
        return logits

    def loss(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self._label_smoothing and self._label_smoothing > 0:
            return F.cross_entropy(logits, target, label_smoothing=self._label_smoothing)
        return F.cross_entropy(logits, target)


# ==================== AdvancedHybridFusion (v2) ====================

class AdaptiveConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, pool: int, dropout: float = 0.1):
        super().__init__()
        padding = kernel_size // 2
        
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm1d(out_channels)
        
        # 残差连接
        self.res_conv = None
        if in_channels != out_channels:
            self.res_conv = nn.Conv1d(in_channels, out_channels, 1)
            
        self.conv2 = nn.Conv1d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        self.act = nn.Mish(inplace=True)  # 改用Mish激活函数
        self.pool = nn.MaxPool1d(pool)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.res_conv is not None:
            identity = self.res_conv(identity)
            
        out += identity  # 残差连接
        out = self.act(out)
        out = self.pool(out)
        out = self.drop(out)
        
        return out


class EnhancedSEBlock(nn.Module):
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        reduced_channels = max(channels // reduction, 8)  # 确保至少8个通道
        
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(channels, reduced_channels),
            nn.Mish(inplace=True),
            nn.Linear(reduced_channels, channels),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attention_weights = self.attention(x).unsqueeze(-1)
        return x * attention_weights


class MultiScaleCNNBackbone(nn.Module):
    def __init__(self, in_channels: int = 1, base_channels: int = 64, out_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        
        # 多尺度初始卷积
        self.init_conv1 = nn.Conv1d(in_channels, base_channels//2, 15, padding=7)
        self.init_conv2 = nn.Conv1d(in_channels, base_channels//2, 31, padding=15)
        self.init_bn = nn.BatchNorm1d(base_channels)
        
        # 主干网络
        self.block1 = AdaptiveConvBlock(base_channels, base_channels, kernel_size=7, pool=2, dropout=dropout)
        self.se1 = EnhancedSEBlock(base_channels)
        
        self.block2 = AdaptiveConvBlock(base_channels, base_channels*2, kernel_size=5, pool=2, dropout=dropout)
        self.se2 = EnhancedSEBlock(base_channels*2)
        
        self.block3 = AdaptiveConvBlock(base_channels*2, base_channels*4, kernel_size=3, pool=2, dropout=dropout)
        self.se3 = EnhancedSEBlock(base_channels*4)
        
        self.block4 = AdaptiveConvBlock(base_channels*4, base_channels*8, kernel_size=3, pool=2, dropout=dropout)
        self.se4 = EnhancedSEBlock(base_channels*8)
        
        # 全局特征提取
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.gmp = nn.AdaptiveMaxPool1d(1)
        
        # 投影层
        self.proj = nn.Sequential(
            nn.Linear(base_channels*8 * 2, 512),  # 平均池化和最大池化拼接
            nn.Mish(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, out_dim),
            nn.LayerNorm(out_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 多尺度输入处理
        x1 = self.init_conv1(x)
        x2 = self.init_conv2(x)
        x = torch.cat([x1, x2], dim=1)
        x = self.init_bn(x)
        x = F.mish(x)
        
        # 主干网络
        x = self.block1(x)
        x = self.se1(x)
        
        x = self.block2(x)
        x = self.se2(x)
        
        x = self.block3(x)
        x = self.se3(x)
        
        x = self.block4(x)
        x = self.se4(x)
        
        # 双池化
        avg_pool = self.gap(x).squeeze(-1)
        max_pool = self.gmp(x).squeeze(-1)
        x = torch.cat([avg_pool, max_pool], dim=1)
        
        x = self.proj(x)
        return x


class EnhancedFeatureMLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 256, out_dim: int = 128, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.Mish(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.Mish(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),
            nn.LayerNorm(out_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class AttentionFusion(nn.Module):
    def __init__(self, cnn_dim: int, feat_dim: int, hidden_dim: int):
        super().__init__()
        self.cnn_proj = nn.Linear(cnn_dim, hidden_dim)
        self.feat_proj = nn.Linear(feat_dim, hidden_dim)
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=8, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, cnn_features: torch.Tensor, feat_features: torch.Tensor) -> torch.Tensor:
        cnn_proj = self.cnn_proj(cnn_features).unsqueeze(1)
        feat_proj = self.feat_proj(feat_features).unsqueeze(1)
        
        # 将两个特征序列拼接
        combined = torch.cat([cnn_proj, feat_proj], dim=1)
        
        # 自注意力融合
        attended, _ = self.attention(combined, combined, combined)
        attended = self.norm(attended)
        
        # 取平均作为融合特征
        fused = attended.mean(dim=1)
        return fused


class AdvancedHybridFusion(nn.Module):
    def __init__(self, feat_in_dim: int, num_classes: int = 5, cnn_out: int = 256, 
                 feat_out: int = 128, fusion_hidden: int = 384, label_smoothing: float = 0.1):
        super().__init__()
        
        # 增强的主干网络
        self.cnn = MultiScaleCNNBackbone(in_channels=1, base_channels=64, out_dim=cnn_out, dropout=0.1)
        self.mlp = EnhancedFeatureMLP(in_dim=feat_in_dim, hidden=256, out_dim=feat_out, dropout=0.2)
        
        # 注意力融合模块
        self.attention_fusion = AttentionFusion(cnn_out, feat_out, fusion_hidden//2)
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(fusion_hidden//2, fusion_hidden//4),
            nn.Mish(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(fusion_hidden//4, num_classes)
        )
        
        # 辅助分类器 - 用于多任务学习
        self.aux_classifier = nn.Linear(cnn_out, num_classes)
        
        self._label_smoothing = label_smoothing
        
        # 初始化权重
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv1d):
            nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, raw_1d: torch.Tensor, feat_vec: torch.Tensor) -> torch.Tensor:
        # CNN特征提取
        f_cnn = self.cnn(raw_1d)
        
        # 手工特征处理
        f_feat = self.mlp(feat_vec)
        
        # 注意力融合
        fused = self.attention_fusion(f_cnn, f_feat)
        
        # 主分类
        logits = self.classifier(fused)
        
        # 辅助分类（仅训练时使用）
        if self.training:
            aux_logits = self.aux_classifier(f_cnn)
            return logits, aux_logits
        
        return logits

    def loss(self, logits: tuple, target: torch.Tensor, aux_weight: float = 0.3) -> torch.Tensor:
        if self.training:
            main_logits, aux_logits = logits
            main_loss = F.cross_entropy(main_logits, target, label_smoothing=self._label_smoothing)
            aux_loss = F.cross_entropy(aux_logits, target, label_smoothing=self._label_smoothing)
            return main_loss + aux_weight * aux_loss
        else:
            return F.cross_entropy(logits, target, label_smoothing=self._label_smoothing)




