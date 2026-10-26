"""
鲁棒损失函数：处理极端值和右偏态分布
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class WeightedMSELoss(nn.Module):
    """加权MSE损失：降低极端值的权重"""
    
    def __init__(self, weight_mode='inverse', alpha=1.0):
        """
        Args:
            weight_mode: 'inverse' 或 'sqrt'
            alpha: 权重强度，越大对极端值惩罚越小
        """
        super().__init__()
        self.weight_mode = weight_mode
        self.alpha = alpha
    
    def forward(self, pred, target):
        """
        Args:
            pred: (B, T, H, W, C)
            target: (B, T, H, W, C)
        """
        if self.weight_mode == 'inverse':
            # 值越大，权重越小
            weights = 1.0 / (1.0 + self.alpha * target.abs())
        elif self.weight_mode == 'sqrt':
            weights = 1.0 / (1.0 + self.alpha * target.abs().sqrt())
        else:
            weights = torch.ones_like(target)
        
        loss = (pred - target) ** 2
        weighted_loss = (loss * weights).mean()
        return weighted_loss


class HuberLossWithMask(nn.Module):
    """Huber损失：对极端值鲁棒，支持NaN掩码"""
    
    def __init__(self, delta=1.0):
        """
        Args:
            delta: Huber损失的阈值，控制对极端值的敏感度
        """
        super().__init__()
        self.delta = delta
    
    def forward(self, pred, target):
        """
        Args:
            pred: (B, T, H, W, C)
            target: (B, T, H, W, C)
        """
        # 创建有效值掩码（非NaN）
        valid_mask = ~torch.isnan(target)
        
        if valid_mask.sum() == 0:
            return torch.tensor(0.0, device=pred.device)
        
        # 只计算有效位置的损失
        pred_valid = pred[valid_mask]
        target_valid = target[valid_mask]
        
        # Huber损失
        error = pred_valid - target_valid
        abs_error = error.abs()
        
        quadratic = torch.clamp(abs_error, max=self.delta)
        linear = abs_error - quadratic
        
        loss = 0.5 * quadratic ** 2 + self.delta * linear
        return loss.mean()


class QuantileLoss(nn.Module):
    """分位数损失：关注不同分位数的预测"""
    
    def __init__(self, quantiles=[0.1, 0.5, 0.9]):
        """
        Args:
            quantiles: 要关注的分位数列表
        """
        super().__init__()
        self.quantiles = quantiles
    
    def forward(self, pred, target):
        """
        Args:
            pred: (B, T, H, W, C)
            target: (B, T, H, W, C)
        """
        losses = []
        for q in self.quantiles:
            error = target - pred
            loss = torch.max(q * error, (q - 1) * error)
            losses.append(loss.mean())
        
        return sum(losses) / len(losses)


class AdaptiveLoss(nn.Module):
    """自适应损失：根据样本难度动态调整权重"""
    
    def __init__(self, base_loss='mse', percentile_threshold=90):
        """
        Args:
            base_loss: 基础损失函数 'mse' 或 'huber'
            percentile_threshold: 高变异性样本的阈值（百分位数）
        """
        super().__init__()
        self.base_loss = base_loss
        self.percentile_threshold = percentile_threshold
        
        if base_loss == 'huber':
            self.loss_fn = nn.HuberLoss(reduction='none', delta=1.0)
        else:
            self.loss_fn = nn.MSELoss(reduction='none')
    
    def forward(self, pred, target):
        """
        Args:
            pred: (B, T, H, W, C)
            target: (B, T, H, W, C)
        """
        B = pred.shape[0]
        
        # 计算每个样本的变异性（标准差）
        target_flat = target.view(B, -1)
        variability = target_flat.std(dim=1)  # (B,)
        
        # 计算阈值
        threshold = torch.quantile(variability, self.percentile_threshold / 100.0)
        
        # 高变异性样本降低权重
        weights = torch.where(
            variability > threshold,
            torch.tensor(0.5, device=pred.device),  # 高变异性样本权重0.5
            torch.tensor(1.0, device=pred.device)   # 普通样本权重1.0
        )
        
        # 计算损失
        loss = self.loss_fn(pred, target)
        loss = loss.view(B, -1).mean(dim=1)  # (B,)
        
        # 加权
        weighted_loss = (loss * weights).mean()
        return weighted_loss


class CombinedLoss(nn.Module):
    """组合损失：MSE + Huber + 加权"""
    
    def __init__(self, mse_weight=0.5, huber_weight=0.3, weighted_weight=0.2):
        super().__init__()
        self.mse_weight = mse_weight
        self.huber_weight = huber_weight
        self.weighted_weight = weighted_weight
        
        self.mse_loss = nn.MSELoss()
        self.huber_loss = HuberLossWithMask(delta=1.0)
        self.weighted_loss = WeightedMSELoss(weight_mode='inverse', alpha=1.0)
    
    def forward(self, pred, target):
        loss = (
            self.mse_weight * self.mse_loss(pred, target) +
            self.huber_weight * self.huber_loss(pred, target) +
            self.weighted_weight * self.weighted_loss(pred, target)
        )
        return loss
