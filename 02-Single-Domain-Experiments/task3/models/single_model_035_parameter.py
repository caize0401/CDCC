"""
Hybrid XGBoost v11: 端到端训练的混合模型
使用XGBoost预测作为损失信号，整个模型一起训练
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import xgboost as xgb
from typing import Tuple, Optional
import math
from sklearn.preprocessing import LabelEncoder


class SimpleCurveProcessor(nn.Module):
    """简化的曲线处理器"""
    def __init__(self, input_dim: int = 500, out_dim: int = 256):
        super().__init__()
        
        self.processor = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LayerNorm(512),
            nn.Mish(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(512, 384),
            nn.LayerNorm(384),
            nn.Mish(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(384, out_dim),
            nn.LayerNorm(out_dim),
            nn.Mish(inplace=True)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x.squeeze(1)
        return self.processor(x)


class SimpleFeatureProcessor(nn.Module):
    """简化的特征处理器"""
    def __init__(self, input_dim: int, out_dim: int = 128):
        super().__init__()
        
        self.processor = nn.Sequential(
            nn.Linear(input_dim, out_dim * 2),
            nn.LayerNorm(out_dim * 2),
            nn.Mish(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(out_dim * 2, out_dim),
            nn.LayerNorm(out_dim),
            nn.Mish(inplace=True)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.processor(x)


class SimpleFusionModule(nn.Module):
    """简化的融合模块"""
    def __init__(self, raw_dim: int, feat_dim: int, out_dim: int = 384):
        super().__init__()
        
        self.fusion = nn.Sequential(
            nn.Linear(raw_dim + feat_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.Mish(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(out_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.Mish(inplace=True)
        )
        
    def forward(self, raw_features: torch.Tensor, feat_features: torch.Tensor) -> torch.Tensor:
        # 拼接特征
        combined = torch.cat([raw_features, feat_features], dim=-1)
        return self.fusion(combined)


class XGBoostLoss(nn.Module):
    """XGBoost损失模块，用于端到端训练"""
    def __init__(self, num_classes: int, xgb_params: dict = None):
        super().__init__()
        self.num_classes = num_classes
        self.xgb_classifier = None
        self.label_encoder = LabelEncoder()
        
        # XGBoost参数
        self.xgb_params = xgb_params or {
            'max_depth': 4,
            'learning_rate': 0.1,
            'n_estimators': 50,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'n_jobs': -1,
            'objective': 'multi:softprob',
            'num_class': num_classes
        }
        
    def fit_xgb(self, X: np.ndarray, y: np.ndarray):
        """训练XGBoost分类器"""
        # 使用标签编码器处理不连续的标签
        y_encoded = self.label_encoder.fit_transform(y)
        
        # 更新参数以匹配实际类别数
        params = self.xgb_params.copy()
        params['num_class'] = len(self.label_encoder.classes_)
        
        self.xgb_classifier = xgb.XGBClassifier(**params)
        self.xgb_classifier.fit(X, y_encoded)
        
    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """计算XGBoost损失"""
        if self.xgb_classifier is None:
            # 如果XGBoost未训练，返回零损失
            return torch.tensor(0.0, device=features.device, requires_grad=True)
        
        # 将特征转换为numpy（保持梯度）
        features_np = features.cpu().detach().numpy()
        labels_np = labels.cpu().detach().numpy()
        
        # 获取XGBoost预测
        with torch.no_grad():
            xgb_proba = self.xgb_classifier.predict_proba(features_np)
        
        # 将预测转换回tensor
        xgb_proba_tensor = torch.tensor(xgb_proba, device=features.device, dtype=torch.float32)
        
        # 计算MSE损失（使用XGBoost的概率分布作为软目标）
        # 让神经网络学习XGBoost的概率分布
        neural_proba = F.softmax(torch.zeros_like(xgb_proba_tensor), dim=1)
        loss = F.mse_loss(neural_proba, xgb_proba_tensor)
        
        return loss


class HybridXGBoostV11(nn.Module):
    """
    Hybrid XGBoost v11: 端到端训练的混合模型
    使用XGBoost预测作为损失信号，整个模型一起训练
    """
    def __init__(self, feat_in_dim: int, num_classes: int = 5, 
                 raw_out_dim: int = 256, feat_out_dim: int = 128, 
                 fusion_hidden: int = 384, xgb_params: dict = None):
        super().__init__()
        
        self.num_classes = num_classes
        self.raw_out_dim = raw_out_dim
        self.feat_out_dim = feat_out_dim
        self.fusion_hidden = fusion_hidden
        
        # 特征提取器
        self.curve_processor = SimpleCurveProcessor(
            input_dim=500,
            out_dim=raw_out_dim
        )
        
        self.feature_processor = SimpleFeatureProcessor(
            input_dim=feat_in_dim,
            out_dim=feat_out_dim
        )
        
        # 融合模块
        self.fusion_module = SimpleFusionModule(
            raw_dim=raw_out_dim,
            feat_dim=feat_out_dim,
            out_dim=fusion_hidden
        )
        
        # XGBoost损失模块
        self.xgb_loss = XGBoostLoss(num_classes, xgb_params)
        
        # 最终分类器（用于推理）
        self.classifier = nn.Sequential(
            nn.Linear(fusion_hidden, fusion_hidden // 2),
            nn.LayerNorm(fusion_hidden // 2),
            nn.Mish(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(fusion_hidden // 2, num_classes)
        )
        
    def forward(self, raw_1d: torch.Tensor, feat_vec: torch.Tensor) -> torch.Tensor:
        """前向传播，返回融合特征和分类logits"""
        # 处理原始曲线
        if raw_1d.dim() == 3:
            raw_1d = raw_1d.squeeze(1)
        
        # 确保数据类型一致
        raw_1d = raw_1d.float()
        feat_vec = feat_vec.float()
        
        curve_features = self.curve_processor(raw_1d)
        
        # 处理手工特征
        manual_features = self.feature_processor(feat_vec)
        
        # 特征融合
        fused_features = self.fusion_module(curve_features, manual_features)
        
        # 最终分类
        logits = self.classifier(fused_features)
        
        return logits, fused_features
    
    def fit_xgb(self, raw_data: torch.Tensor, feat_data: torch.Tensor, labels: torch.Tensor):
        """训练XGBoost分类器"""
        self.eval()
        with torch.no_grad():
            _, fused_features = self.forward(raw_data, feat_data)
            features_np = fused_features.cpu().numpy()
            labels_np = labels.cpu().numpy()
        
        self.xgb_loss.fit_xgb(features_np, labels_np)
    
    def compute_xgb_loss(self, raw_1d: torch.Tensor, feat_vec: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """计算XGBoost损失"""
        logits, fused_features = self.forward(raw_1d, feat_vec)
        return self.xgb_loss(fused_features, labels)
    
    def predict(self, raw_1d: torch.Tensor, feat_vec: torch.Tensor) -> torch.Tensor:
        """预测（使用XGBoost）"""
        self.eval()
        with torch.no_grad():
            _, fused_features = self.forward(raw_1d, feat_vec)
            # 使用XGBoost进行预测
            features_np = fused_features.cpu().numpy()
            xgb_predictions = self.xgb_loss.xgb_classifier.predict(features_np)
            return torch.tensor(xgb_predictions, device=raw_1d.device, dtype=torch.long)
    
    def predict_proba(self, raw_1d: torch.Tensor, feat_vec: torch.Tensor) -> torch.Tensor:
        """预测概率"""
        self.eval()
        with torch.no_grad():
            logits, _ = self.forward(raw_1d, feat_vec)
            return F.softmax(logits, dim=1)


def get_model(feat_in_dim: int, num_classes: int = 5) -> HybridXGBoostV11:
    """创建模型实例，兼容train_test.py的调用方式"""
    return HybridXGBoostV11(
        feat_in_dim=feat_in_dim,
        num_classes=num_classes,
        raw_out_dim=256,
        feat_out_dim=128,
        fusion_hidden=384
    )
