"""
Domain Adversarial Fusion Model v7 with EMA (open version)
复制自 domain_adversarial_fusion_v7.py，供 open 版训练脚本独立使用
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Optional, Dict, List


class GatedResidualBlock(nn.Module):
    """门控残差块"""
    
    def __init__(self, dim: int, expansion: int = 2, dropout: float = 0.1):
        super().__init__()
        
        hidden_dim = dim * expansion
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
        
        self.gate = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Sigmoid()
        )
        
        self.norm = nn.LayerNorm(dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.net(x)
        gate_weights = self.gate(x)
        return self.norm(residual + gate_weights * out)


class MultiHeadFeatureInteraction(nn.Module):
    """多头特征交互模块"""
    
    def __init__(self, dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.dim = dim
        
        # 简化为MLP结构以确保稳定性
        self.interaction = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.GELU()
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.interaction(x)


class EnhancedCurveProcessor(nn.Module):
    """增强的曲线处理器"""
    
    def __init__(self, input_dim: int = 500, base_dim: int = 512, out_dim: int = 256):
        super().__init__()
        
        # 多尺度卷积
        self.conv_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(1, base_dim // 4, kernel_size=3, padding=1),
                nn.BatchNorm1d(base_dim // 4),
                nn.GELU(),
                nn.MaxPool1d(kernel_size=2)
            ),
            nn.Sequential(
                nn.Conv1d(1, base_dim // 4, kernel_size=5, padding=2),
                nn.BatchNorm1d(base_dim // 4),
                nn.GELU(),
                nn.MaxPool1d(kernel_size=2)
            ),
            nn.Sequential(
                nn.Conv1d(1, base_dim // 4, kernel_size=7, padding=3),
                nn.BatchNorm1d(base_dim // 4),
                nn.GELU(),
                nn.MaxPool1d(kernel_size=2)
            ),
            nn.Sequential(
                nn.Conv1d(1, base_dim // 4, kernel_size=9, padding=4),
                nn.BatchNorm1d(base_dim // 4),
                nn.GELU(),
                nn.MaxPool1d(kernel_size=2)
            )
        ])
        
        # 特征融合
        self.fusion = nn.Sequential(
            nn.Linear(base_dim, base_dim),
            nn.LayerNorm(base_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(base_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU()
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x.squeeze(1)
        
        # 多尺度特征提取
        features = []
        for conv in self.conv_layers:
            features.append(conv(x.unsqueeze(1)))
        
        # 拼接特征
        combined = torch.cat(features, dim=1)
        
        # 全局平均池化
        pooled = F.adaptive_avg_pool1d(combined, 1).squeeze(-1)
        
        # 特征融合
        return self.fusion(pooled)


class AdvancedFeatureProcessor(nn.Module):
    """高级特征处理器"""
    
    def __init__(self, in_dim: int, hidden_dims: list = [256, 512, 256], out_dim: int = 128):
        super().__init__()
        
        layers = []
        prev_dim = in_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(0.1),
                GatedResidualBlock(hidden_dim)
            ])
            prev_dim = hidden_dim
        
        # 最终输出层
        layers.extend([
            nn.Linear(prev_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU()
        ])
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class ProgressiveFusionModule(nn.Module):
    """渐进式融合模块"""
    
    def __init__(self, raw_dim: int, feat_dim: int, hidden_dim: int, num_stages: int = 3):
        super().__init__()
        
        self.num_stages = num_stages
        self.fusion_stages = nn.ModuleList()
        
        current_raw_dim = raw_dim
        current_feat_dim = feat_dim
        
        # 创建渐进融合阶段
        for i in range(num_stages):
            stage_hidden = hidden_dim // (2 ** (num_stages - 1 - i))
            
            fusion_stage = nn.Sequential(
                nn.Linear(current_raw_dim + current_feat_dim, stage_hidden),
                nn.LayerNorm(stage_hidden),
                nn.GELU(),
                nn.Dropout(0.1),
                GatedResidualBlock(stage_hidden)
            )
            
            self.fusion_stages.append(fusion_stage)
            current_raw_dim = stage_hidden // 2
            current_feat_dim = stage_hidden // 2
        
        # 最终融合
        self.final_fusion = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            GatedResidualBlock(hidden_dim)
        )
        
    def forward(self, raw_features: torch.Tensor, feat_features: torch.Tensor) -> torch.Tensor:
        # 初始拼接
        fused = torch.cat([raw_features, feat_features], dim=-1)
        
        # 渐进融合
        for stage in self.fusion_stages:
            fused = stage(fused)
        
        # 最终融合
        return self.final_fusion(fused)


def compute_mmd_loss(x: torch.Tensor, y: torch.Tensor, sigmas: List[float] = [0.1, 1.0, 10.0]) -> torch.Tensor:
    """
    计算Maximum Mean Discrepancy (MMD) 损失
    
    Args:
        x: 源域特征 (batch_size, feature_dim)
        y: 目标域特征 (batch_size, feature_dim)
        sigmas: 高斯核的带宽参数列表
    
    Returns:
        MMD损失值
    """
    if x.size(0) == 0 or y.size(0) == 0:
        return torch.tensor(0.0, device=x.device, requires_grad=True)
    
    # 计算核矩阵
    def gaussian_kernel(x, y, sigma):
        pairwise_dist = torch.cdist(x, y, p=2) ** 2
        return torch.exp(-pairwise_dist / (2 * sigma ** 2))
    
    # 计算MMD
    xx = torch.mean(torch.stack([gaussian_kernel(x, x, sigma) for sigma in sigmas]))
    yy = torch.mean(torch.stack([gaussian_kernel(y, y, sigma) for sigma in sigmas]))
    xy = torch.mean(torch.stack([gaussian_kernel(x, y, sigma) for sigma in sigmas]))
    
    return xx + yy - 2 * xy


class EMAModel:
    """EMA模型包装器，用于平滑权重（不继承nn.Module避免循环引用）"""
    
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        
        # 初始化EMA权重
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    def update(self):
        """更新EMA权重"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                # 确保shadow权重在相同设备上
                if self.shadow[name].device != param.device:
                    self.shadow[name] = self.shadow[name].to(param.device)
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()
    
    def apply_shadow(self):
        """应用EMA权重到模型"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                # 确保shadow权重在相同设备上
                if self.shadow[name].device != param.device:
                    self.shadow[name] = self.shadow[name].to(param.device)
                self.backup[name] = param.data
                param.data = self.shadow[name]
    
    def restore(self):
        """恢复原始权重"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                # 确保backup权重在相同设备上
                if self.backup[name].device != param.device:
                    self.backup[name] = self.backup[name].to(param.device)
                param.data = self.backup[name]
        self.backup = {}


class DomainAdversarialFusionV7(nn.Module):
    """
    域对抗融合模型 v7 with EMA
    基于v3模型，引入EMA机制提高泛化能力
    """
    
    def __init__(self, feat_in_dim: int, num_classes: int = 5, num_domains: int = 2,
                 raw_out_dim: int = 256, feat_out_dim: int = 128, 
                 fusion_hidden: int = 384, label_smoothing: float = 0.1,
                 ema_decay: float = 0.999):
        super().__init__()
        
        self.num_classes = num_classes
        self.num_domains = num_domains
        self._label_smoothing = label_smoothing
        self.ema_decay = ema_decay
        
        # 特征提取器
        self.curve_processor = EnhancedCurveProcessor(
            input_dim=500,
            base_dim=512,
            out_dim=raw_out_dim
        )
        
        self.feature_processor = AdvancedFeatureProcessor(
            in_dim=feat_in_dim,
            hidden_dims=[256, 512, 256],
            out_dim=feat_out_dim
        )
        
        # 渐进式融合模块
        self.fusion_module = ProgressiveFusionModule(
            raw_dim=raw_out_dim,
            feat_dim=feat_out_dim,
            hidden_dim=fusion_hidden,
            num_stages=3
        )
        
        # 主分类器
        self.label_classifier = nn.Sequential(
            nn.Linear(fusion_hidden, fusion_hidden // 2),
            nn.LayerNorm(fusion_hidden // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(fusion_hidden // 2, num_classes)
        )
        
        # 域分类器（用于对抗训练）
        self.domain_classifier = nn.Sequential(
            nn.Linear(fusion_hidden, fusion_hidden // 2),
            nn.LayerNorm(fusion_hidden // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(fusion_hidden // 2, num_domains)
        )
        
        # 辅助分类器
        self.aux_classifier = nn.Sequential(
            nn.Linear(raw_out_dim, raw_out_dim // 2),
            nn.LayerNorm(raw_out_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(raw_out_dim // 2, num_classes)
        )
        
        # EMA模型
        self.ema_model = EMAModel(self, decay=ema_decay)
        
        self._init_weights()
    
    def _init_weights(self):
        """权重初始化"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, raw_1d: torch.Tensor, feat_vec: torch.Tensor, 
                alpha: float = 1.0, return_features: bool = False) -> Tuple[torch.Tensor, ...]:
        """
        前向传播
        
        Args:
            raw_1d: 原始曲线数据 (batch_size, 1, seq_len) 或 (batch_size, seq_len)
            feat_vec: 手工特征 (batch_size, feat_dim)
            alpha: 梯度反转强度（用于兼容性，MMD版本不使用）
            return_features: 是否返回中间特征
        
        Returns:
            label_logits: 标签分类结果
            domain_logits: 域分类结果（用于兼容性）
            aux_logits: 辅助分类结果 (训练时)
            features: 融合特征 (如果return_features=True)
        """
        # 处理输入维度
        if raw_1d.dim() == 3:
            raw_1d = raw_1d.squeeze(1)  # (batch_size, seq_len)
        
        # 特征提取
        raw_features = self.curve_processor(raw_1d)
        feat_features = self.feature_processor(feat_vec)
        
        # 特征融合
        fused_features = self.fusion_module(raw_features, feat_features)
        
        # 分类
        label_logits = self.label_classifier(fused_features)
        domain_logits = self.domain_classifier(fused_features)
        
        if return_features:
            # 辅助分类（训练时）
            if self.training:
                aux_logits = self.aux_classifier(raw_features)
                return label_logits, domain_logits, aux_logits, fused_features
            else:
                return label_logits, domain_logits, None, fused_features
        else:
            # 辅助分类（训练时）
            if self.training:
                aux_logits = self.aux_classifier(raw_features)
                return label_logits, domain_logits, aux_logits
            else:
                return label_logits, domain_logits, None
    
    def compute_loss(self, outputs: Tuple[torch.Tensor, ...], 
                    label_targets: torch.Tensor, domain_targets: torch.Tensor,
                    source_features: torch.Tensor, target_features: torch.Tensor,
                    lambda_mmd: float = 1.0, lambda_aux: float = 0.3) -> Tuple[torch.Tensor, Dict]:
        """
        计算包含MMD损失的总损失
        
        Args:
            outputs: 模型输出 (label_logits, domain_logits, aux_logits)
            label_targets: 标签目标
            domain_targets: 域目标
            source_features: 源域特征
            target_features: 目标域特征
            lambda_mmd: MMD损失权重
            lambda_aux: 辅助损失权重
        
        Returns:
            total_loss: 总损失
            loss_dict: 各项损失的字典
        """
        label_logits, domain_logits, aux_logits = outputs
        
        # 标签分类损失（只计算有标签的样本）
        labeled_mask = label_targets != -1
        if labeled_mask.any():
            label_loss = F.cross_entropy(
                label_logits[labeled_mask], 
                label_targets[labeled_mask], 
                label_smoothing=self._label_smoothing
            )
        else:
            label_loss = torch.tensor(0.0, device=label_logits.device)
        
        # MMD损失 - 直接最小化源域和目标域特征分布差异
        mmd_loss = compute_mmd_loss(source_features, target_features)
        
        # 辅助损失（只计算有标签的样本）
        aux_loss = torch.tensor(0.0, device=label_logits.device)
        if aux_logits is not None and labeled_mask.any():
            aux_loss = F.cross_entropy(
                aux_logits[labeled_mask], 
                label_targets[labeled_mask],
                label_smoothing=self._label_smoothing
            )
        
        # 总损失
        total_loss = label_loss + lambda_mmd * mmd_loss + lambda_aux * aux_loss
        
        loss_dict = {
            'total_loss': total_loss.item(),
            'label_loss': label_loss.item(),
            'mmd_loss': mmd_loss.item(),
            'aux_loss': aux_loss.item()
        }
        
        return total_loss, loss_dict
    
    def update_ema(self):
        """更新EMA权重"""
        self.ema_model.update()
    
    def apply_ema(self):
        """应用EMA权重"""
        self.ema_model.apply_shadow()
    
    def restore_ema(self):
        """恢复原始权重"""
        self.ema_model.restore()


def get_model(feat_in_dim: int, num_classes: int = 5) -> DomainAdversarialFusionV7:
    """创建模型实例，兼容train_test.py的调用方式"""
    return DomainAdversarialFusionV7(
        feat_in_dim=feat_in_dim,
        num_classes=num_classes,
        raw_out_dim=256,
        feat_out_dim=128,
        fusion_hidden=384,
        ema_decay=0.999
    )


