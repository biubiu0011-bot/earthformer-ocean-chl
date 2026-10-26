"""
时序数据增强模块
专门为气象/时序预测任务设计的数据增强策略
"""

import numpy as np
import torch
from typing import Tuple, Optional


class TemporalAugmentation:
    """
    时序数据增强类
    
    适用于形状为 (T, H, W, C) 的时序数据
    保持时序特性的同时增加数据多样性
    """
    
    def __init__(
        self,
        enable_noise=True,
        enable_scaling=True,
        enable_spatial_shift=True,
        enable_temporal_shift=False,
        enable_mixup=False,
        noise_std=0.02,
        scale_range=(0.95, 1.05),
        shift_range=2,
        mixup_alpha=0.2,
        augment_prob=0.5
    ):
        """
        Args:
            enable_noise: 是否添加高斯噪声
            enable_scaling: 是否进行数值缩放
            enable_spatial_shift: 是否进行空间平移
            enable_temporal_shift: 是否进行时间偏移（慎用）
            enable_mixup: 是否使用 Mixup
            noise_std: 高斯噪声标准差
            scale_range: 缩放范围 (min, max)
            shift_range: 空间平移的最大像素数
            mixup_alpha: Mixup 的 alpha 参数
            augment_prob: 每种增强的应用概率
        """
        self.enable_noise = enable_noise
        self.enable_scaling = enable_scaling
        self.enable_spatial_shift = enable_spatial_shift
        self.enable_temporal_shift = enable_temporal_shift
        self.enable_mixup = enable_mixup
        
        self.noise_std = noise_std
        self.scale_range = scale_range
        self.shift_range = shift_range
        self.mixup_alpha = mixup_alpha
        self.augment_prob = augment_prob
    
    def __call__(self, seq: np.ndarray, target: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        对输入序列和目标序列应用数据增强
        
        Args:
            seq: 输入序列，形状 (T, H, W, C)
            target: 目标序列，形状 (T, H, W, C)
        
        Returns:
            增强后的 (seq, target)
        """
        # 1. 高斯噪声增强
        if self.enable_noise and np.random.rand() < self.augment_prob:
            seq, target = self._add_gaussian_noise(seq, target)
        
        # 2. 数值缩放增强
        if self.enable_scaling and np.random.rand() < self.augment_prob:
            seq, target = self._scale_values(seq, target)
        
        # 3. 空间平移增强
        if self.enable_spatial_shift and np.random.rand() < self.augment_prob:
            seq, target = self._spatial_shift(seq, target)
        
        # 4. 时间偏移增强（慎用，可能破坏时序关系）
        if self.enable_temporal_shift and np.random.rand() < self.augment_prob:
            seq = self._temporal_shift(seq)
            target = self._temporal_shift(target)
        
        return seq, target
    
    def _add_gaussian_noise(self, seq: np.ndarray, target: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        添加高斯噪声
        模拟测量误差和数据不确定性
        """
        noise_seq = np.random.normal(0, self.noise_std, seq.shape).astype(np.float32)
        noise_target = np.random.normal(0, self.noise_std, target.shape).astype(np.float32)
        
        seq_aug = seq + noise_seq
        target_aug = target + noise_target
        
        return seq_aug, target_aug
    
    def _scale_values(self, seq: np.ndarray, target: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        数值缩放
        模拟不同强度的气象现象
        """
        # 为每个通道生成不同的缩放因子
        scale_seq = np.random.uniform(
            self.scale_range[0], 
            self.scale_range[1], 
            size=(1, 1, 1, seq.shape[-1])
        ).astype(np.float32)
        
        scale_target = np.random.uniform(
            self.scale_range[0], 
            self.scale_range[1], 
            size=(1, 1, 1, target.shape[-1])
        ).astype(np.float32)
        
        seq_aug = seq * scale_seq
        target_aug = target * scale_target
        
        return seq_aug, target_aug
    
    def _spatial_shift(self, seq: np.ndarray, target: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        空间平移
        模拟气象系统的空间位置变化
        """
        shift_h = np.random.randint(-self.shift_range, self.shift_range + 1)
        shift_w = np.random.randint(-self.shift_range, self.shift_range + 1)
        
        seq_aug = np.roll(seq, shift=(shift_h, shift_w), axis=(1, 2))
        target_aug = np.roll(target, shift=(shift_h, shift_w), axis=(1, 2))
        
        return seq_aug, target_aug
    
    def _temporal_shift(self, data: np.ndarray) -> np.ndarray:
        """
        时间偏移（慎用）
        轻微调整时间步
        """
        shift_t = np.random.randint(-1, 2)  # -1, 0, 1
        if shift_t != 0:
            data_aug = np.roll(data, shift=shift_t, axis=0)
        else:
            data_aug = data
        
        return data_aug


class DistributionAdaptiveAugmentation:
    """
    分布自适应增强
    根据训练集和测试集的统计差异进行针对性增强
    """
    
    def __init__(
        self,
        train_stats: Optional[dict] = None,
        test_stats: Optional[dict] = None,
        adapt_mean=True,
        adapt_std=True,
        adapt_prob=0.3
    ):
        """
        Args:
            train_stats: 训练集统计信息 {'mean': array, 'std': array}
            test_stats: 测试集统计信息 {'mean': array, 'std': array}
            adapt_mean: 是否调整均值
            adapt_std: 是否调整标准差
            adapt_prob: 应用概率
        """
        self.train_stats = train_stats
        self.test_stats = test_stats
        self.adapt_mean = adapt_mean
        self.adapt_std = adapt_std
        self.adapt_prob = adapt_prob
    
    def __call__(self, seq: np.ndarray, target: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        应用分布自适应增强
        """
        if self.train_stats is None or self.test_stats is None:
            return seq, target
        
        if np.random.rand() > self.adapt_prob:
            return seq, target
        
        # 对输入序列进行分布调整
        seq_aug = self._adapt_distribution(seq, 'input')
        target_aug = self._adapt_distribution(target, 'output')
        
        return seq_aug, target_aug
    
    def _adapt_distribution(self, data: np.ndarray, data_type: str) -> np.ndarray:
        """
        调整数据分布使其更接近测试集
        """
        train_mean = self.train_stats[data_type]['mean']
        train_std = self.train_stats[data_type]['std']
        test_mean = self.test_stats[data_type]['mean']
        test_std = self.test_stats[data_type]['std']
        
        # 标准化到 N(0, 1)
        data_normalized = (data - train_mean) / (train_std + 1e-8)
        
        # 重新缩放到测试集分布
        # 使用插值，不完全转换到测试集分布
        alpha = np.random.uniform(0.3, 0.7)  # 插值系数
        
        target_mean = train_mean * (1 - alpha) + test_mean * alpha
        target_std = train_std * (1 - alpha) + test_std * alpha
        
        data_adapted = data_normalized * target_std + target_mean
        
        return data_adapted.astype(np.float32)


class MixupAugmentation:
    """
    Mixup 数据增强
    混合两个样本来创建新的训练样本
    """
    
    def __init__(self, alpha=0.2, prob=0.3):
        """
        Args:
            alpha: Beta 分布的参数
            prob: 应用 Mixup 的概率
        """
        self.alpha = alpha
        self.prob = prob
    
    def __call__(
        self, 
        seq1: np.ndarray, 
        target1: np.ndarray,
        seq2: np.ndarray,
        target2: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        对两个样本应用 Mixup
        
        Returns:
            混合后的 (seq, target)
        """
        if np.random.rand() > self.prob:
            return seq1, target1
        
        # 从 Beta 分布采样混合系数
        lam = np.random.beta(self.alpha, self.alpha)
        
        # 混合
        seq_mixed = lam * seq1 + (1 - lam) * seq2
        target_mixed = lam * target1 + (1 - lam) * target2
        
        return seq_mixed, target_mixed


def compute_dataset_statistics(dataloader, device='cpu'):
    """
    计算数据集的统计信息
    
    Args:
        dataloader: PyTorch DataLoader
        device: 计算设备
    
    Returns:
        dict: {'input': {'mean': array, 'std': array}, 
               'output': {'mean': array, 'std': array}}
    """
    input_sum = None
    input_sq_sum = None
    output_sum = None
    output_sq_sum = None
    n_samples = 0
    
    print("正在计算数据集统计信息...")
    
    for batch_idx, (seq, target) in enumerate(dataloader):
        if isinstance(seq, torch.Tensor):
            seq = seq.numpy()
        if isinstance(target, torch.Tensor):
            target = target.numpy()
        
        batch_size = seq.shape[0]
        n_samples += batch_size
        
        # 计算输入的统计信息
        if input_sum is None:
            input_sum = seq.sum(axis=(0, 1, 2, 3))
            input_sq_sum = (seq ** 2).sum(axis=(0, 1, 2, 3))
            output_sum = target.sum(axis=(0, 1, 2, 3))
            output_sq_sum = (target ** 2).sum(axis=(0, 1, 2, 3))
        else:
            input_sum += seq.sum(axis=(0, 1, 2, 3))
            input_sq_sum += (seq ** 2).sum(axis=(0, 1, 2, 3))
            output_sum += target.sum(axis=(0, 1, 2, 3))
            output_sq_sum += (target ** 2).sum(axis=(0, 1, 2, 3))
        
        if (batch_idx + 1) % 100 == 0:
            print(f"  已处理 {batch_idx + 1} 个批次...")
    
    # 计算均值和标准差
    n_elements_input = n_samples * seq.shape[1] * seq.shape[2] * seq.shape[3]
    n_elements_output = n_samples * target.shape[1] * target.shape[2] * target.shape[3]
    
    input_mean = input_sum / n_elements_input
    input_std = np.sqrt(input_sq_sum / n_elements_input - input_mean ** 2)
    
    output_mean = output_sum / n_elements_output
    output_std = np.sqrt(output_sq_sum / n_elements_output - output_mean ** 2)
    
    stats = {
        'input': {
            'mean': input_mean.astype(np.float32),
            'std': input_std.astype(np.float32)
        },
        'output': {
            'mean': output_mean.astype(np.float32),
            'std': output_std.astype(np.float32)
        }
    }
    
    print("✓ 统计信息计算完成")
    print(f"  输入均值: {input_mean}")
    print(f"  输入标准差: {input_std}")
    print(f"  输出均值: {output_mean}")
    print(f"  输出标准差: {output_std}")
    
    return stats
