"""
HybridFusion Pro: 高级混合模型
结合门控机制、深度交互和渐进式融合策略
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class GatedResidualBlock(nn.Module):
    """门控残差块，提供更好的梯度流和特征选择"""
    def __init__(self, dim: int, expansion: int = 2, dropout: float = 0.1):
        super().__init__()
        
        hidden_dim = dim * expansion
        
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Mish(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
        
        # 门控机制
        self.gate = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Sigmoid()
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.net(x)
        gate_weights = self.gate(x)
        out = out * gate_weights
        return residual + out


class MultiHeadFeatureInteraction(nn.Module):
    """多头特征交互模块"""
    def __init__(self, dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        
        # 查询、键、值投影
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        
        # 输出投影
        self.out_proj = nn.Linear(dim, dim)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, dim = x.shape
        
        # 投影到Q, K, V
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 注意力计算
        attn_weights = (q @ k.transpose(-2, -1)) * self.scale
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # 注意力应用
        attn_output = (attn_weights @ v).transpose(1, 2).contiguous().view(batch_size, seq_len, dim)
        
        # 输出投影
        output = self.out_proj(attn_output)
        
        return output


class EnhancedCurveProcessor(nn.Module):
    """增强的曲线处理器，结合局部和全局特征"""
    def __init__(self, input_dim: int = 500, base_dim: int = 512, out_dim: int = 256):
        super().__init__()
        
        # 多尺度特征提取
        self.scale_branches = nn.ModuleList([
            # 局部特征分支
            nn.Sequential(
                nn.Linear(input_dim, base_dim),
                nn.LayerNorm(base_dim),
                nn.Mish(inplace=True),
                nn.Dropout(0.2),
                GatedResidualBlock(base_dim),
            ),
            # 全局特征分支  
            nn.Sequential(
                nn.Linear(input_dim, base_dim),
                nn.LayerNorm(base_dim),
                nn.Mish(inplace=True),
                nn.Dropout(0.2),
                GatedResidualBlock(base_dim),
            )
        ])
        
        # 特征交互
        self.feature_interaction = MultiHeadFeatureInteraction(base_dim * 2)
        
        # 特征精炼
        self.refinement = nn.Sequential(
            GatedResidualBlock(base_dim * 2),
            GatedResidualBlock(base_dim * 2),
            nn.Linear(base_dim * 2, out_dim),
            nn.LayerNorm(out_dim),
            nn.Mish(inplace=True)
        )
        
        # 特征重要性权重
        self.feature_weights = nn.Parameter(torch.ones(2))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 多尺度特征提取
        branch_outputs = []
        for branch in self.scale_branches:
            branch_outputs.append(branch(x))
        
        # 加权融合
        weighted_outputs = []
        for i, output in enumerate(branch_outputs):
            weight = torch.sigmoid(self.feature_weights[i])
            weighted_outputs.append(output * weight)
        
        # 拼接特征
        combined = torch.cat(weighted_outputs, dim=-1)
        
        # 特征交互 (添加序列维度)
        combined_seq = combined.unsqueeze(1)
        interacted = self.feature_interaction(combined_seq).squeeze(1)
        
        # 特征精炼
        refined = self.refinement(interacted)
        
        return refined


class AdvancedFeatureProcessor(nn.Module):
    """高级特征处理器，用于手工特征"""
    def __init__(self, in_dim: int, hidden_dims: list = [256, 512, 256], out_dim: int = 128):
        super().__init__()
        
        layers = []
        prev_dim = in_dim
        
        # 构建深度特征提取网络
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.Mish(inplace=True),
                nn.Dropout(0.2),
                GatedResidualBlock(hidden_dim)
            ])
            prev_dim = hidden_dim
        
        # 输出层
        layers.extend([
            nn.Linear(prev_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.Mish(inplace=True)
        ])
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class ProgressiveFusionModule(nn.Module):
    """渐进式融合模块，逐步融合两种模态特征"""
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
                nn.Mish(inplace=True),
                nn.Dropout(0.1),
                GatedResidualBlock(stage_hidden)
            )
            
            self.fusion_stages.append(fusion_stage)
            
            # 更新维度用于下一阶段
            current_raw_dim = stage_hidden // 2
            current_feat_dim = stage_hidden // 2
    
    def forward(self, raw_features: torch.Tensor, feat_features: torch.Tensor) -> torch.Tensor:
        # 初始拼接
        fused = torch.cat([raw_features, feat_features], dim=-1)
        
        # 渐进融合
        for stage in self.fusion_stages:
            fused = stage(fused)
        
        return fused


class HybridFusionPro(nn.Module):
    """
    HybridFusion Pro: 高级混合模型
    特征：
    - 门控残差块提供更好的训练稳定性
    - 多头特征交互增强特征表达能力
    - 渐进式融合策略优化特征组合
    - 多尺度曲线处理捕捉不同粒度特征
    """
    def __init__(self, feat_in_dim: int, num_classes: int = 5, 
                 raw_out_dim: int = 256, feat_out_dim: int = 128, 
                 fusion_hidden: int = 384, label_smoothing: float = 0.1):
        super().__init__()
        
        # 原始曲线处理器
        self.curve_processor = EnhancedCurveProcessor(
            input_dim=500,
            base_dim=512,
            out_dim=raw_out_dim
        )
        
        # 手工特征处理器
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
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(fusion_hidden, fusion_hidden // 2),
            nn.LayerNorm(fusion_hidden // 2),
            nn.Mish(inplace=True),
            nn.Dropout(0.3),
            GatedResidualBlock(fusion_hidden // 2),
            
            nn.Linear(fusion_hidden // 2, fusion_hidden // 4),
            nn.LayerNorm(fusion_hidden // 4),
            nn.Mish(inplace=True),
            nn.Dropout(0.2),
            GatedResidualBlock(fusion_hidden // 4),
            
            nn.Linear(fusion_hidden // 4, num_classes)
        )
        
        # 多个辅助分类器
        self.aux_classifiers = nn.ModuleList([
            nn.Linear(raw_out_dim, num_classes),
            nn.Linear(feat_out_dim, num_classes)
        ])
        
        self._label_smoothing = label_smoothing
        
        # 初始化
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def forward(self, raw_1d: torch.Tensor, feat_vec: torch.Tensor) -> torch.Tensor:
        # 处理原始曲线
        if raw_1d.dim() == 3:
            raw_1d = raw_1d.squeeze(1)
        
        curve_features = self.curve_processor(raw_1d)
        
        # 处理手工特征
        manual_features = self.feature_processor(feat_vec)
        
        # 特征融合
        fused_features = self.fusion_module(curve_features, manual_features)
        
        # 主分类
        main_logits = self.classifier(fused_features)
        
        # 辅助分类（训练时）
        if self.training:
            aux_logits = [
                self.aux_classifiers[0](curve_features),
                self.aux_classifiers[1](manual_features)
            ]
            return main_logits, aux_logits
        
        return main_logits
    
    def loss(self, logits, target: torch.Tensor, aux_weights: list = [0.2, 0.1]) -> torch.Tensor:
        if self.training and isinstance(logits, tuple):
            main_logits, aux_logits_list = logits
            
            main_loss = F.cross_entropy(main_logits, target, label_smoothing=self._label_smoothing)
            
            aux_loss = 0
            for i, aux_logits in enumerate(aux_logits_list):
                aux_loss += aux_weights[i] * F.cross_entropy(
                    aux_logits, target, label_smoothing=self._label_smoothing
                )
            
            return main_loss + aux_loss
        else:
            if isinstance(logits, tuple):
                logits = logits[0]
            return F.cross_entropy(logits, target, label_smoothing=self._label_smoothing)


# 测试代码
if __name__ == "__main__":
    # 模拟输入
    batch_size = 32
    seq_len = 500
    feat_dim = 35
    
    raw_input = torch.randn(batch_size, 1, seq_len)
    feat_input = torch.randn(batch_size, feat_dim)
    target = torch.randint(0, 5, (batch_size,))
    
    # 创建模型
    model = HybridFusionPro(feat_in_dim=feat_dim, num_classes=5)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"总参数数量: {total_params:,}")
    print(f"可训练参数: {trainable_params:,}")
    
    # 训练模式测试
    model.train()
    logits = model(raw_input, feat_input)
    if isinstance(logits, tuple):
        main_logits, aux_logits = logits
        print(f"训练模式 - 主输出: {main_logits.shape}, 辅助输出: {len(aux_logits)}个")
        loss = model.loss(logits, target)
        print(f"训练损失: {loss.item():.4f}")
    
    # 推理模式测试
    model.eval()
    with torch.no_grad():
        logits = model(raw_input, feat_input)
        print(f"推理模式 - 输出形状: {logits.shape}")
        preds = torch.argmax(logits, dim=1)
        accuracy = (preds == target).float().mean()
        print(f"测试准确率: {accuracy.item():.4f}")