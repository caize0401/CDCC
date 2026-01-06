"""
Domain Adversarial Fusion Model with MMD Loss
基于MMD损失的域对抗融合模型 v3
通过MMD直接最小化源域和目标域特征分布之间的差异
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
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
            nn.Dropout(dropout)
        )
        
        self.norm = nn.LayerNorm(dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        output = self.interaction(x)
        return self.norm(residual + output)


class EnhancedCurveProcessor(nn.Module):
    """增强的曲线处理器"""
    
    def __init__(self, input_dim: int = 500, base_dim: int = 512, out_dim: int = 256):
        super().__init__()
        
        # 多尺度特征提取
        self.scale_processors = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(1, base_dim // 4, kernel_size=k, padding=k//2),
                nn.BatchNorm1d(base_dim // 4),
                nn.GELU(),
                nn.AdaptiveAvgPool1d(64)
            ) for k in [3, 5, 7, 9]
        ])
        
        # 计算实际的拼接特征维度
        # 4个尺度 × (base_dim // 4) × 64 = base_dim × 64
        self.concat_dim = base_dim * 64
        
        # 特征融合
        self.feature_fusion = nn.Sequential(
            nn.Linear(self.concat_dim, out_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(out_dim * 2, out_dim)
        )
        
        # 多头交互
        self.interaction = MultiHeadFeatureInteraction(out_dim, num_heads=8)
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 多尺度特征提取
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (batch_size, 1, seq_len)
        
        scale_features = []
        for processor in self.scale_processors:
            scale_feat = processor(x)  # (batch_size, channels, 64)
            scale_features.append(scale_feat.flatten(1))
        
        # 拼接多尺度特征
        combined = torch.cat(scale_features, dim=1)
        
        # 特征融合和交互
        processed = self.feature_fusion(combined)
        enhanced = self.interaction(processed)
        
        return enhanced


class AdvancedFeatureProcessor(nn.Module):
    """高级特征处理器"""
    
    def __init__(self, in_dim: int, hidden_dims: list = [256, 512, 256], out_dim: int = 128):
        super().__init__()
        
        layers = []
        prev_dim = in_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.GELU(),
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, out_dim))
        
        self.processor = nn.Sequential(*layers)
        self.interaction = MultiHeadFeatureInteraction(out_dim, num_heads=4)
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        processed = self.processor(x)
        enhanced = self.interaction(processed)
        return enhanced


class ProgressiveFusionModule(nn.Module):
    """渐进式融合模块"""
    
    def __init__(self, raw_dim: int, feat_dim: int, hidden_dim: int, num_stages: int = 3):
        super().__init__()
        
        # 初始投影到相同维度
        self.raw_proj = nn.Linear(raw_dim, hidden_dim)
        self.feat_proj = nn.Linear(feat_dim, hidden_dim)
        
        # 门控残差融合
        self.fusion_layers = nn.ModuleList([
            GatedResidualBlock(hidden_dim, expansion=2) 
            for _ in range(num_stages)
        ])
        
        self.final_norm = nn.LayerNorm(hidden_dim)
    
    def forward(self, raw_features: torch.Tensor, feat_features: torch.Tensor) -> torch.Tensor:
        # 投影到相同维度
        raw_proj = self.raw_proj(raw_features)
        feat_proj = self.feat_proj(feat_features)
        
        # 初始融合
        fused = raw_proj + feat_proj
        
        # 渐进式融合
        for fusion_layer in self.fusion_layers:
            fused = fusion_layer(fused)
        
        return self.final_norm(fused)


def gaussian_kernel(x: torch.Tensor, y: torch.Tensor, sigma: float = 1.0) -> torch.Tensor:
    """计算高斯核"""
    x_size = x.size(0)
    y_size = y.size(0)
    dim = x.size(1)
    
    x = x.unsqueeze(1)  # (x_size, 1, dim)
    y = y.unsqueeze(0)  # (1, y_size, dim)
    
    tiled_x = x.expand(x_size, y_size, dim)
    tiled_y = y.expand(x_size, y_size, dim)
    
    kernel_input = (tiled_x - tiled_y).pow(2).mean(2) / float(dim)
    return torch.exp(-kernel_input / (2 * sigma**2))


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
    
    xx_loss = 0
    yy_loss = 0
    xy_loss = 0
    
    for sigma in sigmas:
        kernel_xx = gaussian_kernel(x, x, sigma)
        kernel_yy = gaussian_kernel(y, y, sigma)
        kernel_xy = gaussian_kernel(x, y, sigma)
        
        xx_loss += kernel_xx.mean()
        yy_loss += kernel_yy.mean()
        xy_loss += kernel_xy.mean()
    
    mmd_loss = xx_loss + yy_loss - 2 * xy_loss
    return mmd_loss


class DomainAdversarialFusionMMD(nn.Module):
    """
    基于MMD损失的域对抗融合模型
    通过MMD直接最小化源域和目标域特征分布之间的差异
    """
    
    def __init__(self, feat_in_dim: int, num_classes: int = 5, num_domains: int = 2,
                 raw_out_dim: int = 256, feat_out_dim: int = 128, 
                 fusion_hidden: int = 384, label_smoothing: float = 0.1):
        super().__init__()
        
        self.num_classes = num_classes
        self.num_domains = num_domains
        self._label_smoothing = label_smoothing
        
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
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(fusion_hidden // 2, num_classes)
        )
        
        # 域分类器（用于对抗训练）
        self.domain_classifier = nn.Sequential(
            nn.Linear(fusion_hidden, fusion_hidden // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(fusion_hidden // 2, num_domains)
        )
        
        # 辅助分类器
        self.aux_classifier = nn.Sequential(
            nn.Linear(raw_out_dim, raw_out_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(raw_out_dim // 2, num_classes)
        )
        
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
        
        # 主分类
        label_logits = self.label_classifier(fused_features)
        
        # 域分类（为了兼容性保留，但在MMD版本中不用于损失计算）
        domain_logits = self.domain_classifier(fused_features)
        
        # 辅助分类（仅训练时）
        aux_logits = None
        if self.training:
            aux_logits = self.aux_classifier(raw_features)
        
        if return_features:
            return label_logits, domain_logits, aux_logits, fused_features
        else:
            return label_logits, domain_logits, aux_logits
    
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
            'aux_loss': aux_loss.item(),
        }
        
        return total_loss, loss_dict
    
    def predict_labels(self, raw_1d: torch.Tensor, feat_vec: torch.Tensor) -> torch.Tensor:
        """预测标签概率"""
        self.eval()
        with torch.no_grad():
            outputs = self(raw_1d, feat_vec)
            label_logits = outputs[0]
            return F.softmax(label_logits, dim=1)

    def compute_loss_with_adv(self, outputs: Tuple[torch.Tensor, ...],
                              label_targets: torch.Tensor, domain_targets: torch.Tensor,
                              source_features: torch.Tensor, target_features: torch.Tensor,
                              lambda_mmd: float = 1.0, lambda_aux: float = 0.3, lambda_adv: float = 0.1) -> Tuple[torch.Tensor, Dict]:
        """
        含对抗项的总损失（与训练脚本兼容）。

        说明：当前模型为MMD主导的版本，没有显式GRL层；域分类器作为兼容保留。
        这里实现的对抗项为域分类交叉熵，权重由 lambda_adv 控制；
        若不需对抗，可将 lambda_adv 设为 0。
        """
        label_logits, domain_logits, aux_logits = outputs

        # 先计算基础损失（标签 + MMD + 辅助）
        base_total_loss, loss_dict = self.compute_loss(
            (label_logits, domain_logits, aux_logits),
            label_targets, domain_targets,
            source_features, target_features,
            lambda_mmd=lambda_mmd, lambda_aux=lambda_aux
        )

        # 域分类损失（整批都有域标签）
        adv_loss = F.cross_entropy(domain_logits, domain_targets)

        total_loss = base_total_loss + lambda_adv * adv_loss

        # 补充字典信息
        loss_dict.update({
            'adv_loss': adv_loss.item(),
            'total_loss': total_loss.item(),
        })

        return total_loss, loss_dict