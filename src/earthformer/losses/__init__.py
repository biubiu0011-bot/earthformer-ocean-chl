"""
Robust loss functions for handling extreme values and skewed distributions.
"""

from .robust_losses import (
    HuberLossWithMask,
    WeightedMSELoss,
    QuantileLoss,
    AdaptiveLoss,
    CombinedLoss,
)

__all__ = [
    'HuberLossWithMask',
    'WeightedMSELoss',
    'QuantileLoss',
    'AdaptiveLoss',
    'CombinedLoss',
]
