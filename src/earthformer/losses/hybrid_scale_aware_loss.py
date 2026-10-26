"""
Hybrid Scale-aware Loss
混合尺度感知损失函数

基于 ChatGPT 的建议实现：
- 在 log-z 空间（归一化空间）中计算损失
- 结合 Huber Loss（鲁棒性）和 MSE（整体偏差）
- 可选：对低浓度区域额外加权（避免 MAPE 爆炸）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class HybridScaleAwareLoss(nn.Module):
    """
    混合尺度感知损失函数
    
    L = α · Huber(y, ŷ) + (1 - α) · MSE(y, ŷ)
    
    其中 y, ŷ 都在 log-z 空间（归一化空间）
    
    参数:
        alpha: Huber Loss 的权重，推荐 0.6-0.8
        delta: Huber Loss 的阈值，推荐 1.0
        use_weighted_mse: 是否对低浓度区域加权
        weight_epsilon: 权重计算的 epsilon，避免除零
    """
    
    def __init__(self, alpha=0.9, delta=2.0, use_weighted_mse=False, weight_epsilon=1.0):
        super().__init__()
        self.alpha = alpha
        self.delta = delta
        self.use_weighted_mse = use_weighted_mse
        self.weight_epsilon = weight_epsilon
        
        print(f"\n{'='*60}")
        print(f"初始化 Hybrid Scale-aware Loss")
        print(f"{'='*60}")
        print(f"  alpha (Huber权重): {self.alpha}")
        print(f"  delta (Huber阈值): {self.delta}")
        print(f"  use_weighted_mse: {self.use_weighted_mse}")
        if self.use_weighted_mse:
            print(f"  weight_epsilon: {self.weight_epsilon}")
        print(f"{'='*60}\n")
    
    def forward(self, pred, target, mask=None):
        """
        前向传播
        
        参数:
            pred: 预测值，shape (B, T, H, W, C) 或 (B, C, T, H, W)
            target: 真实值，shape 同 pred
            mask: 可选的掩码，shape 同 pred
        
        返回:
            loss: 标量损失值
        """
        # 1. Huber Loss（鲁棒性，对异常值不敏感）
        huber_loss = F.huber_loss(pred, target, delta=self.delta, reduction='none')
        
        # 2. MSE Loss（整体偏差）
        if self.use_weighted_mse:
            # 对低浓度区域加权
            # w = 1 / (|y| + ε)
            # 这样可以避免 MAPE 爆炸，同时关注小值
            weights = 1.0 / (torch.abs(target) + self.weight_epsilon)
            
            # 归一化权重（保持损失的尺度）
            weights = weights / weights.mean()
            
            # 加权 MSE
            mse_loss = (pred - target) ** 2 * weights
        else:
            # 标准 MSE
            mse_loss = (pred - target) ** 2
        
        # 3. 应用掩码（如果提供）
        if mask is not None:
            huber_loss = huber_loss * mask
            mse_loss = mse_loss * mask
            
            # 计算有效元素的平均值
            num_valid = mask.sum()
            if num_valid > 0:
                huber_loss = huber_loss.sum() / num_valid
                mse_loss = mse_loss.sum() / num_valid
            else:
                huber_loss = huber_loss.mean()
                mse_loss = mse_loss.mean()
        else:
            huber_loss = huber_loss.mean()
            mse_loss = mse_loss.mean()
        
        # 4. 组合损失
        total_loss = self.alpha * huber_loss + (1 - self.alpha) * mse_loss
        
        return total_loss
    
    def __repr__(self):
        return (f"HybridScaleAwareLoss(alpha={self.alpha}, delta={self.delta}, "
                f"use_weighted_mse={self.use_weighted_mse})")


class AdaptiveHybridLoss(nn.Module):
    """
    自适应混合损失函数
    
    根据训练进度动态调整 alpha：
    - 训练初期：更多 MSE（快速收敛）
    - 训练后期：更多 Huber（鲁棒性）
    
    参数:
        alpha_start: 初始 alpha 值
        alpha_end: 最终 alpha 值
        warmup_epochs: 预热轮数
        delta: Huber Loss 的阈值
    """
    
    def __init__(self, alpha_start=0.5, alpha_end=0.8, warmup_epochs=50, delta=1.0):
        super().__init__()
        self.alpha_start = alpha_start
        self.alpha_end = alpha_end
        self.warmup_epochs = warmup_epochs
        self.delta = delta
        self.current_epoch = 0
        
        print(f"\n{'='*60}")
        print(f"初始化 Adaptive Hybrid Loss")
        print(f"{'='*60}")
        print(f"  alpha_start: {self.alpha_start}")
        print(f"  alpha_end: {self.alpha_end}")
        print(f"  warmup_epochs: {self.warmup_epochs}")
        print(f"  delta: {self.delta}")
        print(f"{'='*60}\n")
    
    def set_epoch(self, epoch):
        """设置当前 epoch"""
        self.current_epoch = epoch
    
    def get_alpha(self):
        """计算当前的 alpha 值"""
        if self.current_epoch >= self.warmup_epochs:
            return self.alpha_end
        
        # 线性插值
        progress = self.current_epoch / self.warmup_epochs
        alpha = self.alpha_start + (self.alpha_end - self.alpha_start) * progress
        return alpha
    
    def forward(self, pred, target, mask=None):
        """前向传播"""
        alpha = self.get_alpha()
        
        # Huber Loss
        huber_loss = F.huber_loss(pred, target, delta=self.delta, reduction='none')
        
        # MSE Loss
        mse_loss = (pred - target) ** 2
        
        # 应用掩码
        if mask is not None:
            huber_loss = huber_loss * mask
            mse_loss = mse_loss * mask
            
            num_valid = mask.sum()
            if num_valid > 0:
                huber_loss = huber_loss.sum() / num_valid
                mse_loss = mse_loss.sum() / num_valid
            else:
                huber_loss = huber_loss.mean()
                mse_loss = mse_loss.mean()
        else:
            huber_loss = huber_loss.mean()
            mse_loss = mse_loss.mean()
        
        # 组合损失
        total_loss = alpha * huber_loss + (1 - alpha) * mse_loss
        
        return total_loss


class MultiScaleLoss(nn.Module):
    """
    多尺度损失函数
    
    在不同的尺度上计算损失：
    1. 归一化空间（log-z 空间）
    2. log 空间（反 Z-score 后）
    3. 原始空间（完全反归一化后）
    
    这样可以同时优化不同空间的误差
    
    参数:
        alpha_normalized: 归一化空间的权重
        alpha_log: log 空间的权重
        alpha_original: 原始空间的权重
        denormalize_fn: 反归一化函数
    """
    
    def __init__(self, alpha_normalized=0.5, alpha_log=0.3, alpha_original=0.2,
                 denormalize_fn=None, delta=1.0):
        super().__init__()
        self.alpha_normalized = alpha_normalized
        self.alpha_log = alpha_log
        self.alpha_original = alpha_original
        self.denormalize_fn = denormalize_fn
        self.delta = delta
        
        # 归一化权重
        total = self.alpha_normalized + self.alpha_log + self.alpha_original
        self.alpha_normalized /= total
        self.alpha_log /= total
        self.alpha_original /= total
        
        print(f"\n{'='*60}")
        print(f"初始化 Multi-Scale Loss")
        print(f"{'='*60}")
        print(f"  归一化空间权重: {self.alpha_normalized:.3f}")
        print(f"  log 空间权重: {self.alpha_log:.3f}")
        print(f"  原始空间权重: {self.alpha_original:.3f}")
        print(f"{'='*60}\n")
    
    def forward(self, pred, target, mask=None):
        """前向传播"""
        # 1. 归一化空间的损失
        loss_normalized = F.huber_loss(pred, target, delta=self.delta, reduction='mean')
        
        total_loss = self.alpha_normalized * loss_normalized
        
        # 2. log 空间的损失（如果提供了反归一化函数）
        if self.denormalize_fn is not None and self.alpha_log > 0:
            # 反 Z-score 到 log 空间
            pred_log = self.denormalize_fn(pred, to_log_space=True)
            target_log = self.denormalize_fn(target, to_log_space=True)
            
            loss_log = F.l1_loss(pred_log, target_log, reduction='mean')
            total_loss += self.alpha_log * loss_log
        
        # 3. 原始空间的损失
        if self.denormalize_fn is not None and self.alpha_original > 0:
            # 完全反归一化到原始空间
            pred_original = self.denormalize_fn(pred, to_log_space=False)
            target_original = self.denormalize_fn(target, to_log_space=False)
            
            loss_original = F.l1_loss(pred_original, target_original, reduction='mean')
            total_loss += self.alpha_original * loss_original
        
        return total_loss


# ========== 便捷函数 ==========

def get_hybrid_loss(loss_type='hybrid', **kwargs):
    """
    获取混合损失函数
    
    参数:
        loss_type: 损失函数类型
            - 'hybrid': HybridScaleAwareLoss（推荐）
            - 'adaptive': AdaptiveHybridLoss
            - 'multiscale': MultiScaleLoss
        **kwargs: 传递给损失函数的参数
    
    返回:
        损失函数实例
    """
    if loss_type == 'hybrid':
        return HybridScaleAwareLoss(**kwargs)
    elif loss_type == 'adaptive':
        return AdaptiveHybridLoss(**kwargs)
    elif loss_type == 'multiscale':
        return MultiScaleLoss(**kwargs)
    else:
        raise ValueError(f"未知的损失函数类型: {loss_type}")


if __name__ == '__main__':
    # 测试
    print("测试 Hybrid Scale-aware Loss")
    
    # 创建测试数据
    pred = torch.randn(2, 10, 45, 60, 20)
    target = torch.randn(2, 10, 45, 60, 20)
    
    # 测试 HybridScaleAwareLoss
    print("\n1. HybridScaleAwareLoss")
    loss_fn = HybridScaleAwareLoss(alpha=0.7, delta=1.0)
    loss = loss_fn(pred, target)
    print(f"Loss: {loss.item():.6f}")
    
    # 测试加权 MSE
    print("\n2. HybridScaleAwareLoss with weighted MSE")
    loss_fn = HybridScaleAwareLoss(alpha=0.7, delta=1.0, use_weighted_mse=True)
    loss = loss_fn(pred, target)
    print(f"Loss: {loss.item():.6f}")
    
    # 测试 AdaptiveHybridLoss
    print("\n3. AdaptiveHybridLoss")
    loss_fn = AdaptiveHybridLoss(alpha_start=0.5, alpha_end=0.8, warmup_epochs=50)
    for epoch in [0, 25, 50, 100]:
        loss_fn.set_epoch(epoch)
        loss = loss_fn(pred, target)
        print(f"Epoch {epoch}: alpha={loss_fn.get_alpha():.3f}, loss={loss.item():.6f}")
    
    print("\n✅ 测试完成")
