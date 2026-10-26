"""Residual Blocks for Spatiotemporal Modeling"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ImprovedSpatioTemporalResBlock(nn.Module):
    """
    改进的时空残差块 - 结合标准 ResNet 和时空分解的优点
    
    特点：
    1. 支持通道变换（in_channels -> out_channels）
    2. 支持下采样（stride）
    3. 可选 3D 卷积或 2D+1D 分解（默认分解，更快）
    4. 标准的 shortcut 连接
    """
    
    def __init__(self, in_channels, out_channels, stride=1, use_3d=False, kernel_size=3):
        """
        Args:
            in_channels: 输入通道数
            out_channels: 输出通道数
            stride: 步长（用于下采样）
            use_3d: True=使用3D卷积（简单但慢），False=使用2D+1D分解（快3-5倍）
            kernel_size: 卷积核大小
        """
        super().__init__()
        self.use_3d = use_3d
        self.stride = stride
        
        if use_3d:
            # 3D 卷积版本（标准 ResNet 风格）
            padding = kernel_size // 2
            self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size, 
                                  stride=stride, padding=padding, bias=False)
            self.bn1 = nn.BatchNorm3d(out_channels)
            self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size, 
                                  stride=1, padding=padding, bias=False)
            self.bn2 = nn.BatchNorm3d(out_channels)
        else:
            # 2D+1D 分解版本（更高效）
            spatial_padding = kernel_size // 2
            temporal_padding = kernel_size // 2
            
            # 空间卷积（处理 H, W 维度）
            self.spatial_conv1 = nn.Conv2d(in_channels, out_channels, kernel_size,
                                          stride=stride, padding=spatial_padding, bias=False)
            self.spatial_bn1 = nn.BatchNorm2d(out_channels)
            
            # 时间卷积（处理 T 维度）
            self.temporal_conv1 = nn.Conv1d(out_channels, out_channels, kernel_size,
                                           padding=temporal_padding, bias=False)
            self.temporal_bn1 = nn.BatchNorm1d(out_channels)
            
            # 第二组卷积
            self.spatial_conv2 = nn.Conv2d(out_channels, out_channels, kernel_size,
                                          padding=spatial_padding, bias=False)
            self.spatial_bn2 = nn.BatchNorm2d(out_channels)
            
            self.temporal_conv2 = nn.Conv1d(out_channels, out_channels, kernel_size,
                                           padding=temporal_padding, bias=False)
            self.temporal_bn2 = nn.BatchNorm1d(out_channels)
        
        self.relu = nn.ReLU(inplace=True)
        
        # Shortcut 连接
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()
    
    def forward(self, x):
        """
        x: (B, T, H, W, C) 或 (B, C, T, H, W)
        返回: 与输入相同格式
        """
        # 转换为 (B, C, T, H, W) 格式
        if x.dim() == 5 and x.shape[1] != x.shape[-1]:
            # 假设输入是 (B, T, H, W, C)
            x = x.permute(0, 4, 1, 2, 3)
            need_permute_back = True
        else:
            need_permute_back = False
        
        # Shortcut
        residual = self.shortcut(x)
        
        if self.use_3d:
            # 3D 卷积路径
            out = self.conv1(x)
            out = self.bn1(out)
            out = self.relu(out)
            
            out = self.conv2(out)
            out = self.bn2(out)
        else:
            # 2D+1D 分解路径
            B, C, T, H, W = x.shape
            
            # 第一组：空间卷积 + 时间卷积
            # 空间卷积
            out = x.permute(0, 2, 1, 3, 4).reshape(B*T, C, H, W)  # (B*T, C, H, W)
            out = self.spatial_conv1(out)
            out = self.spatial_bn1(out)
            out = self.relu(out)
            
            # 获取新的空间尺寸
            _, C_out, H_new, W_new = out.shape
            out = out.reshape(B, T, C_out, H_new, W_new).permute(0, 2, 1, 3, 4)  # (B, C, T, H, W)
            
            # 时间卷积
            out = out.permute(0, 3, 4, 2, 1).reshape(B*H_new*W_new, C_out, T)  # (B*H*W, C, T)
            out = self.temporal_conv1(out)
            out = self.temporal_bn1(out)
            out = self.relu(out)
            out = out.reshape(B, H_new, W_new, C_out, T).permute(0, 3, 4, 1, 2)  # (B, C, T, H, W)
            
            # 第二组：空间卷积 + 时间卷积
            # 空间卷积
            out = out.permute(0, 2, 1, 3, 4).reshape(B*T, C_out, H_new, W_new)
            out = self.spatial_conv2(out)
            out = self.spatial_bn2(out)
            out = out.reshape(B, T, C_out, H_new, W_new).permute(0, 2, 1, 3, 4)
            
            # 时间卷积
            out = out.permute(0, 3, 4, 2, 1).reshape(B*H_new*W_new, C_out, T)
            out = self.temporal_conv2(out)
            out = self.temporal_bn2(out)
            out = out.reshape(B, H_new, W_new, C_out, T).permute(0, 3, 4, 1, 2)
        
        # 残差连接
        out = out + residual
        out = self.relu(out)
        
        # 转换回原始格式
        if need_permute_back:
            out = out.permute(0, 2, 3, 4, 1)
        
        return out


class SpatioTemporalResBlock(nn.Module):
    """时空残差块 - 适合视频/时空序列预测"""
    
    def __init__(self, channels, kernel_size=3, use_3d=True):
        """
        Args:
            channels: 通道数
            kernel_size: 卷积核大小
            use_3d: 是否使用3D卷积（True）还是2D+1D分解（False）
        """
        super().__init__()
        self.use_3d = use_3d
        
        if use_3d:
            # 3D 卷积版本
            padding = kernel_size // 2
            self.conv1 = nn.Conv3d(channels, channels, kernel_size, 
                                  padding=padding, bias=False)
            self.bn1 = nn.BatchNorm3d(channels)
            self.conv2 = nn.Conv3d(channels, channels, kernel_size, 
                                  padding=padding, bias=False)
            self.bn2 = nn.BatchNorm3d(channels)
        else:
            # 2D+1D 分解版本（更高效）
            self.spatial_conv = nn.Conv2d(channels, channels, kernel_size,
                                         padding=kernel_size//2, bias=False)
            self.temporal_conv = nn.Conv1d(channels, channels, kernel_size,
                                          padding=kernel_size//2, bias=False)
            self.bn1 = nn.BatchNorm3d(channels)
            self.bn2 = nn.BatchNorm3d(channels)
        
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        """
        x: (B, C, T, H, W) 或 (B, T, H, W, C)
        """
        # 转换为 (B, C, T, H, W) 格式
        if x.dim() == 5 and x.shape[1] != x.shape[-1]:
            # 假设输入是 (B, T, H, W, C)
            x = x.permute(0, 4, 1, 2, 3)
            need_permute_back = True
        else:
            need_permute_back = False
        
        residual = x
        
        if self.use_3d:
            # 3D 卷积路径
            out = self.conv1(x)
            out = self.bn1(out)
            out = self.relu(out)
            out = self.conv2(out)
            out = self.bn2(out)
        else:
            # 2D+1D 分解路径
            B, C, T, H, W = x.shape
            
            # 空间卷积
            out = x.permute(0, 2, 1, 3, 4).reshape(B*T, C, H, W)
            out = self.spatial_conv(out)
            out = out.reshape(B, T, C, H, W).permute(0, 2, 1, 3, 4)
            out = self.bn1(out)
            out = self.relu(out)
            
            # 时间卷积
            out = out.permute(0, 3, 4, 2, 1).reshape(B*H*W, C, T)
            out = self.temporal_conv(out)
            out = out.reshape(B, H, W, C, T).permute(0, 3, 4, 1, 2)
            out = self.bn2(out)
        
        # 残差连接
        out = out + residual
        out = self.relu(out)
        
        # 转换回原始格式
        if need_permute_back:
            out = out.permute(0, 2, 3, 4, 1)
        
        return out


class BottleneckResBlock(nn.Module):
    """瓶颈残差块 - 类似 ResNet-50/101"""
    
    def __init__(self, in_channels, bottleneck_channels=None, expansion=4):
        """
        Args:
            in_channels: 输入通道数
            bottleneck_channels: 瓶颈层通道数（默认为 in_channels // 4）
            expansion: 扩展因子
        """
        super().__init__()
        if bottleneck_channels is None:
            bottleneck_channels = in_channels // 4
        
        out_channels = bottleneck_channels * expansion
        
        # 1x1 降维
        self.conv1 = nn.Conv3d(in_channels, bottleneck_channels, 1, bias=False)
        self.bn1 = nn.BatchNorm3d(bottleneck_channels)
        
        # 3x3 主卷积
        self.conv2 = nn.Conv3d(bottleneck_channels, bottleneck_channels, 3,
                              padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(bottleneck_channels)
        
        # 1x1 升维
        self.conv3 = nn.Conv3d(bottleneck_channels, out_channels, 1, bias=False)
        self.bn3 = nn.BatchNorm3d(out_channels)
        
        self.relu = nn.ReLU(inplace=True)
        
        # 如果输入输出通道数不同，需要投影
        self.downsample = None
        if in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm3d(out_channels)
            )
    
    def forward(self, x):
        """
        x: (B, T, H, W, C) 或 (B, C, T, H, W)
        """
        # 转换为 (B, C, T, H, W) 格式
        if x.dim() == 5 and x.shape[1] != x.shape[-1]:
            # 假设输入是 (B, T, H, W, C)
            x = x.permute(0, 4, 1, 2, 3)
            need_permute_back = True
        else:
            need_permute_back = False
        
        residual = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        
        out = self.conv3(out)
        out = self.bn3(out)
        
        # 投影残差（如果需要）
        if self.downsample is not None:
            residual = self.downsample(residual)
        
        out = out + residual
        out = self.relu(out)
        
        # 转换回原始格式
        if need_permute_back:
            out = out.permute(0, 2, 3, 4, 1)
        
        return out


class DenseResBlock(nn.Module):
    """密集残差块 - 结合 DenseNet 和 ResNet"""
    
    def __init__(self, in_channels, growth_rate=32, num_layers=4):
        """
        Args:
            in_channels: 输入通道数
            growth_rate: 每层增长的通道数
            num_layers: 层数
        """
        super().__init__()
        self.num_layers = num_layers
        self.growth_rate = growth_rate
        
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            layer_in_channels = in_channels + i * growth_rate
            self.layers.append(
                nn.Sequential(
                    nn.BatchNorm3d(layer_in_channels),
                    nn.ReLU(inplace=True),
                    nn.Conv3d(layer_in_channels, growth_rate, 3, padding=1, bias=False)
                )
            )
        
        # Transition layer
        final_channels = in_channels + num_layers * growth_rate
        self.transition = nn.Sequential(
            nn.BatchNorm3d(final_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(final_channels, in_channels, 1, bias=False)
        )
    
    def forward(self, x):
        """
        x: (B, T, H, W, C) 或 (B, C, T, H, W)
        """
        # 转换为 (B, C, T, H, W) 格式
        if x.dim() == 5 and x.shape[1] != x.shape[-1]:
            # 假设输入是 (B, T, H, W, C)
            x = x.permute(0, 4, 1, 2, 3)
            need_permute_back = True
        else:
            need_permute_back = False
        
        residual = x
        features = [x]
        
        # 密集连接
        for layer in self.layers:
            new_features = layer(torch.cat(features, dim=1))
            features.append(new_features)
        
        # Transition
        out = self.transition(torch.cat(features, dim=1))
        
        # 残差连接
        out = out + residual
        
        # 转换回原始格式
        if need_permute_back:
            out = out.permute(0, 2, 3, 4, 1)
        
        return out


class SEResBlock(nn.Module):
    """带 Squeeze-and-Excitation 的残差块"""
    
    def __init__(self, channels, reduction=16):
        """
        Args:
            channels: 通道数
            reduction: SE 模块的降维比例
        """
        super().__init__()
        
        # 主卷积路径
        self.conv1 = nn.Conv3d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(channels)
        self.conv2 = nn.Conv3d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(channels)
        self.relu = nn.ReLU(inplace=True)
        
        # SE 模块
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(channels, channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv3d(channels // reduction, channels, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        """
        x: (B, T, H, W, C) 或 (B, C, T, H, W)
        """
        # 转换为 (B, C, T, H, W) 格式
        if x.dim() == 5 and x.shape[1] != x.shape[-1]:
            # 假设输入是 (B, T, H, W, C)
            x = x.permute(0, 4, 1, 2, 3)
            need_permute_back = True
        else:
            need_permute_back = False
        
        residual = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        # SE 注意力
        se_weight = self.se(out)
        out = out * se_weight
        
        # 残差连接
        out = out + residual
        out = self.relu(out)
        
        # 转换回原始格式
        if need_permute_back:
            out = out.permute(0, 2, 3, 4, 1)
        
        return out


class ResBlockStack(nn.Module):
    """残差块堆叠 - 可以插入到 Transformer 中"""
    
    def __init__(self, channels, num_blocks=2, block_type='basic', use_3d=False):
        """
        Args:
            channels: 通道数
            num_blocks: 残差块数量
            block_type: 'improved', 'basic', 'bottleneck', 'se', 'dense'
            use_3d: 是否使用3D卷积（仅对 improved 和 basic 有效）
        """
        super().__init__()
        
        blocks = []
        for _ in range(num_blocks):
            if block_type == 'improved':
                # 使用改进版残差块（推荐）
                blocks.append(ImprovedSpatioTemporalResBlock(channels, channels, stride=1, use_3d=use_3d))
            elif block_type == 'basic':
                blocks.append(SpatioTemporalResBlock(channels, use_3d=use_3d))
            elif block_type == 'bottleneck':
                blocks.append(BottleneckResBlock(channels))
            elif block_type == 'se':
                blocks.append(SEResBlock(channels))
            elif block_type == 'dense':
                blocks.append(DenseResBlock(channels))
            else:
                raise ValueError(f"Unknown block_type: {block_type}. Choose from: 'improved', 'basic', 'bottleneck', 'se', 'dense'")
        
        self.blocks = nn.Sequential(*blocks)
    
    def forward(self, x):
        """
        x: (B, T, H, W, C) 或 (B, C, T, H, W)
        """
        return self.blocks(x)


class HybridTransformerResBlock(nn.Module):
    """混合 Transformer + 残差块"""
    
    def __init__(self, channels, num_heads=4):
        """
        结合 Transformer 的全局建模和 CNN 的局部建模
        
        Args:
            channels: 通道数
            num_heads: 注意力头数
        """
        super().__init__()
        
        # CNN 分支（局部特征）
        self.cnn_branch = SpatioTemporalResBlock(channels, use_3d=False)
        
        # Transformer 分支（全局特征）
        self.norm1 = nn.LayerNorm(channels)
        self.attn = nn.MultiheadAttention(channels, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(channels)
        self.ffn = nn.Sequential(
            nn.Linear(channels, channels * 4),
            nn.GELU(),
            nn.Linear(channels * 4, channels)
        )
        
        # 融合权重
        self.alpha = nn.Parameter(torch.tensor(0.5))
    
    def forward(self, x):
        """
        x: (B, T, H, W, C)
        """
        B, T, H, W, C = x.shape
        
        # CNN 分支
        x_cnn = x.permute(0, 4, 1, 2, 3)  # (B, C, T, H, W)
        x_cnn = self.cnn_branch(x_cnn)
        x_cnn = x_cnn.permute(0, 2, 3, 4, 1)  # (B, T, H, W, C)
        
        # Transformer 分支
        x_flat = x.reshape(B, T*H*W, C)
        x_attn = self.norm1(x_flat)
        x_attn, _ = self.attn(x_attn, x_attn, x_attn)
        x_attn = x_attn + x_flat
        
        x_ffn = self.norm2(x_attn)
        x_ffn = self.ffn(x_ffn)
        x_transformer = x_attn + x_ffn
        x_transformer = x_transformer.reshape(B, T, H, W, C)
        
        # 加权融合
        alpha = torch.sigmoid(self.alpha)
        out = alpha * x_cnn + (1 - alpha) * x_transformer
        
        return out
