"""
UACAN: 双输入编码 + 融合 + 分类器 + 条件域对抗 + 类条件 cwMMD + 能量未知检测 + 动态调度
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class GradientReversal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None


def grad_reverse(x: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
    return GradientReversal.apply(x, alpha)


class CurveEncoder(nn.Module):
    """E_c: 曲线编码器 -> F_c"""

    def __init__(self, input_dim: int, out_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(256, out_dim),
            nn.LayerNorm(out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x.squeeze(1)
        return self.net(x)


class FeatureEncoder(nn.Module):
    """E_f: 特征编码器 -> F_f"""

    def __init__(self, input_dim: int, out_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(256, out_dim),
            nn.LayerNorm(out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Fusion(nn.Module):
    """φ(F_c, F_f) -> F 共享表示"""

    def __init__(self, dim_c: int, dim_f: int, out_dim: int = 256):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(dim_c + dim_f, out_dim),
            nn.LayerNorm(out_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
        )
        self._out_dim = out_dim

    def forward(self, F_c: torch.Tensor, F_f: torch.Tensor) -> torch.Tensor:
        x = torch.cat([F_c, F_f], dim=1)
        return self.proj(x)


class UACAN(nn.Module):
    """
    双输入编码 -> 融合 F -> 分类器 C -> P(y|x)
    条件域对抗: z = F ⊗ P(y|x), D(z)
    类条件 MMD + 类别权重 w_k
    能量未知检测: E(x) = -T log Σ_k e^(f_k/T), L_unk 对高能量样本
    """

    def __init__(
        self,
        curve_dim: int,
        feat_dim: int,
        num_classes: int,
        hidden_dim: int = 128,
        fused_dim: int = 256,
        num_domains: int = 2,
        label_smoothing: float = 0.1,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.fused_dim = fused_dim
        self._label_smoothing = label_smoothing

        self.E_c = CurveEncoder(curve_dim, hidden_dim)
        self.E_f = FeatureEncoder(feat_dim, hidden_dim)
        self.fusion = Fusion(hidden_dim, hidden_dim, fused_dim)
        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, fused_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(fused_dim // 2, num_classes),
        )
        # 域判别器输入 z = F ⊗ P，即 F 与 P 的外积/拼接等，这里用拼接 [F, P] 简化实现（与张量积同属条件信息）
        self.domain_disc = nn.Sequential(
            nn.Linear(fused_dim + num_classes, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, num_domains),
        )

    def forward(
        self,
        x_curve: torch.Tensor,
        x_feat: torch.Tensor,
        alpha: float = 0.0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        F_c = self.E_c(x_curve)
        F_f = self.E_f(x_feat)
        F = self.fusion(F_c, F_f)
        logits = self.classifier(F)
        P = F.softmax(logits, dim=1)
        z = torch.cat([F, P], dim=1)
        z_rev = grad_reverse(z, alpha)
        domain_logits = self.domain_disc(z_rev)
        return logits, domain_logits, F

    def energy(self, logits: torch.Tensor, T: float = 1.0) -> torch.Tensor:
        """E(x) = -T * log(Σ_k exp(f_k/T))"""
        return -T * torch.logsumexp(logits / T, dim=1)

    @staticmethod
    def compute_class_centers(
        F: torch.Tensor,
        labels: torch.Tensor,
        num_classes: int,
        probs: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        源域: 按标签硬分配（仅 labels>=0），目标域: 按 probs 软分配。
        Returns: (K, dim)
        """
        device = F.device
        dim = F.size(1)
        centers = torch.zeros(num_classes, dim, device=device, dtype=F.dtype)
        if probs is None:
            for k in range(num_classes):
                mask = (labels == k) & (labels >= 0)
                if mask.any():
                    centers[k] = F[mask].mean(dim=0)
        else:
            for k in range(num_classes):
                w = probs[:, k]
                if w.sum() > 1e-8:
                    centers[k] = (w.unsqueeze(1) * F).sum(dim=0) / (w.sum() + 1e-8)
        return centers

    @staticmethod
    def cwmmd_loss(
        F_s: torch.Tensor,
        F_t: torch.Tensor,
        y_s: torch.Tensor,
        P_t: torch.Tensor,
        num_classes: int,
        w_k: torch.Tensor,
    ) -> torch.Tensor:
        """L_cwMMD = Σ_k w_k ||μ_s^k - μ_t^k||^2"""
        mu_s = UACAN.compute_class_centers(F_s, y_s, num_classes, probs=None)
        mu_t = UACAN.compute_class_centers(F_t, None, num_classes, probs=P_t)
        loss = 0.0
        for k in range(num_classes):
            loss = loss + w_k[k] * ((mu_s[k] - mu_t[k]) ** 2).sum()
        return loss

    def compute_loss(
        self,
        logits_s: torch.Tensor,
        domain_logits_s: torch.Tensor,
        logits_t: torch.Tensor,
        domain_logits_t: torch.Tensor,
        F_s: torch.Tensor,
        F_t: torch.Tensor,
        y_s: torch.Tensor,
        d_s: torch.Tensor,
        d_t: torch.Tensor,
        lambda_domain: float,
        lambda_cwmmd: float,
        lambda_unk: float,
        T: float = 1.0,
        delta: float = 0.0,
        margin: float = 0.0,
    ) -> Tuple[torch.Tensor, dict]:
        """
        L_cls + λ_domain * L_domain + λ_cwMMD * L_cwMMD + λ_unk * L_unk
        """
        device = logits_s.device
        K = self.num_classes

        # 源域分类
        labeled = y_s >= 0
        if labeled.any():
            L_cls = F.cross_entropy(
                logits_s[labeled], y_s[labeled].clamp(0, K - 1),
                label_smoothing=self._label_smoothing,
            )
        else:
            L_cls = torch.tensor(0.0, device=device)

        # 条件域对抗: -E_s log D(z_s) - E_t log(1-D(z_t))
        L_domain = F.cross_entropy(domain_logits_s, d_s) + F.cross_entropy(domain_logits_t, d_t)

        # 类别权重 w_k = (1/N_t) Σ_j P(y=k|x_j^t)
        P_t = F.softmax(logits_t, dim=1)
        w_k = P_t.mean(dim=0)
        # 类条件 MMD（仅对已知 K 类）
        L_cwmmd = self.cwmmd_loss(F_s, F_t, y_s, P_t, K, w_k)

        # 能量与未知拒绝
        E_t = self.energy(logits_t, T)
        high_e = E_t > delta
        if high_e.any():
            L_unk = F.relu(margin - E_t[high_e]).mean()
        else:
            L_unk = torch.tensor(0.0, device=device)

        total = L_cls + lambda_domain * L_domain + lambda_cwmmd * L_cwmmd + lambda_unk * L_unk
        return total, {
            "L_cls": L_cls.item(),
            "L_domain": L_domain.item(),
            "L_cwmmd": L_cwmmd.item(),
            "L_unk": L_unk.item() if isinstance(L_unk, torch.Tensor) else L_unk,
        }
