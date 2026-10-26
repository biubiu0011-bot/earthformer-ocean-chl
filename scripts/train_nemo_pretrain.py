import sys
import os
sys.path.append(os.path.abspath(os.path.dirname(__file__)+"/../../../src/"))
sys.path.append("../..")

# 修复 X11 错误：设置 matplotlib 使用非交互式后端
import matplotlib
matplotlib.use('Agg')  # 必须在 import pyplot 之前设置

import warnings
from shutil import copyfile
import inspect
import numpy as np
import torch
import torch 

from torch import nn
from torch.nn import functional as F
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR
import torchmetrics
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, DeviceStatsMonitor, Callback
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from omegaconf import OmegaConf
# from pytorch_msssim import ssim as ssim_fn
from torchmetrics.functional import structural_similarity_index_measure as ssim_func
from einops import rearrange



import argparse
from einops import rearrange
from pytorch_lightning import Trainer, seed_everything
from earthformer.config import cfg
from earthformer.utils.optim import SequentialLR, warmup_lambda
from earthformer.utils.utils import get_parameter_names
from earthformer.utils.layout import layout_to_in_out_slice
from earthformer.visualization.nbody import save_example_vis_results
from earthformer.cuboid_transformer.cuboid_transformer_update import CuboidTransformerModel
from earthformer_ocean_chl.data.nemo_dataset import MovingMNISTDataModule
from earthformer.utils.apex_ddp import ApexDDPStrategy
from earthformer.losses.robust_losses import HuberLossWithMask, AdaptiveLoss, CombinedLoss
from earthformer.losses.hybrid_scale_aware_loss import HybridScaleAwareLoss, AdaptiveHybridLoss, get_hybrid_loss

# os.environ['cuda_visible_devices']='0'
_curr_dir = os.path.realpath(os.path.dirname(os.path.realpath(__file__)))
exps_dir = os.path.join(_curr_dir, "experiments")

result = []
target_result = []  # 用于保存真实值

# ========== 反归一化和掩码处理函数 ==========

# ========== log10 + Z-score 反归一化 ==========
def output_inver(data, clip_range=None, eps=1e-6):
    """
    log10 + Z-score 反归一化

    正向变换:
        1. log10(x + eps)
        2. (log_x - mean) / std

    反向变换:
        1. log_x = normalized * std + mean
        2. x = 10^log_x - eps

    参数:
        data: 归一化后的数据，shape 可为 (B, C, T, H, W) 或 (B, T, H, W, C)
        clip_range: 可选裁剪范围，用于限制极端值
        eps: 防止 log(0) 时使用的小值
    """

    # ===== 手动导入 mean 和 std =====
    mean_lst = [-0.8099777086420732, -0.8115802958917924, -0.8127624462175829, -0.8151966975596096, 
                -0.8135008989053839, -0.8052269245363659, -0.7867536288753555, -0.7652295936793179, 
                -0.7549894884624446, -0.7622548591216773, -0.7876066983095401, -0.8291325690572037, 
                -0.8828961201219838, -0.9511702408536956, -1.0325219981755032, -1.1183872076021382, 
                -1.290728874146602, -1.4370798880198143, -1.7191277585118485, -1.9407677923512865]

    std_lst = [0.3833945911822738, 0.3933290964825488, 0.4001694684783994, 0.40570293037471034, 
               0.3974837675022117, 0.37002891688461675, 0.3228279917449538, 0.2726069823532655, 
               0.24944933565008104, 0.2606833738126882, 0.29335556564184173, 0.3321853228500587, 
               0.35661859054696915, 0.37289097526152526, 0.38108980051253044, 0.3724381585608597, 
               0.3443756764747817, 0.31354102008419454, 0.24487589830245718, 0.1943395166387652]

    print(f"\n[log10 + Z-score 反归一化]")
    print("="*60)
    print(f"输入形状: {data.shape}")
    print(f"输入范围: [{np.nanmin(data):.6f}, {np.nanmax(data):.6f}]")
    print(f"输入均值: {np.nanmean(data):.6f}, 标准差: {np.nanstd(data):.6f}")

    # 可选：裁剪极端值
    if clip_range is not None:
        data_clipped = np.clip(data, clip_range[0], clip_range[1])
        clipped_count = np.sum((data < clip_range[0]) | (data > clip_range[1]))
        if clipped_count > 0:
            print(f"⚠️ 裁剪了 {clipped_count} 个超出范围 {clip_range} 的值 ({100*clipped_count/data.size:.3f}%)")
            data = data_clipped

    # 检查通道维度，如果最后一维是通道
    if data.shape[-1] == len(mean_lst):
        # (B, T, H, W, C) → (B, C, T, H, W)
        data = np.transpose(data, (0, 4, 1, 2, 3))

    B, C, T, H, W = data.shape
    if C != len(mean_lst):
        print(f"⚠️ 通道数({C})与 mean/std 列表长度({len(mean_lst)})不匹配")
        C = min(C, len(mean_lst))

    # 转换为数组并调整形状
    mean_arr = np.array(mean_lst[:C]).reshape(1, C, 1, 1, 1)
    std_arr = np.array(std_lst[:C]).reshape(1, C, 1, 1, 1)

    # 1️⃣ 反 Z-score → log10 空间
    log_data = data[:, :C] * std_arr + mean_arr
    print(f"\nlog10 空间:")
    print(f"  范围: [{np.nanmin(log_data):.6f}, {np.nanmax(log_data):.6f}]")
    print(f"  均值: {np.nanmean(log_data):.6f}")

    # 2️⃣ 反 log10 → 原始空间
    data_denorm = np.power(10, log_data) - eps
    data_denorm = np.maximum(data_denorm, 0)  # 保证非负

    print(f"\n反对数变换:")
    print(f"  范围: [{np.nanmin(data_denorm):.6f}, {np.nanmax(data_denorm):.6f}]")
    print(f"  均值: {np.nanmean(data_denorm):.6f}")
    print(f"  NaN: {np.isnan(data_denorm).sum()}, Inf: {np.isinf(data_denorm).sum()}")

    # 显示前5个通道的统计信息
    print(f"\n逐通道统计（前5个）:")
    for c in range(min(5, C)):
        channel_data = data_denorm[:, c]
        print(f"  通道 {c:2d}: [{np.nanmin(channel_data):.6f}, {np.nanmax(channel_data):.6f}], "
              f"均值={np.nanmean(channel_data):.6f}")

    # (B, C, T, H, W) → (B, T, H, W, C)
    data_denorm = np.transpose(data_denorm, (0, 2, 3, 4, 1))

    print(f"\n最终结果:")
    print(f"  形状: {data_denorm.shape}")
    print(f"  范围: [{np.nanmin(data_denorm):.6f}, {np.nanmax(data_denorm):.6f}]")
    print(f"  均值: {np.nanmean(data_denorm):.6f}")
    print("="*60+"\n")

    return data_denorm



def set_mask(data, mask_file):
    """
    掩码处理函数
    data: (B, T, H, W, C) = (batch, 14, 45, 60, 20)
    mask_file: (C, T, H, W) = (20, 14, 45, 60)
    """
    print(f"掩码处理前形状: data={data.shape}, mask={mask_file.shape}")
    
    data_length = data.shape[0]
    
    # 从 (B, T, H, W, C) 转换为 (B, C, T, H, W)
    data_transposed = np.transpose(data, (0, 4, 1, 2, 3))  # (B, C, T, H, W)
    
    # 扩展 mask 到 batch 维: (C, T, H, W) -> (B, C, T, H, W)
    mask = np.tile(mask_file, (data_length, 1, 1, 1, 1))
    
    # nan 的位置设为 True（需要 mask 的地方）
    bool_mask = np.isnan(mask)
    data_transposed[bool_mask] = np.nan
    
    # 转换回 (B, T, H, W, C)
    data_masked = np.transpose(data_transposed, (0, 2, 3, 4, 1))
    
    print(f"掩码处理后形状: {data_masked.shape}")
    print(f"掩码处理后NaN数量: {np.isnan(data_masked).sum()} / {data_masked.size} ({100*np.isnan(data_masked).sum()/data_masked.size:.2f}%)")
    
    return data_masked


class Metrics():
    """自定义指标计算类"""
    def __init__(self, test_y, pred):
        self.test_y = test_y
        self.pred = pred
    
    def R2_score(self):
        MSE = np.nanmean(np.power((self.test_y - self.pred), 2))
        var = np.nanvar(self.test_y)
        if var == 0:
            return np.nan
        return 1 - MSE / var
    
    def RMSE_score(self):
        return np.sqrt(np.nanmean(np.power((self.test_y - self.pred), 2)))
    
    def correlation_coef(self):
        test_y1 = self.test_y.flatten()
        pred1 = self.pred.flatten()
        nan_indices = np.isnan(test_y1) | np.isnan(pred1)
        test_y1 = test_y1[~nan_indices]
        pred1 = pred1[~nan_indices]
        
        if len(test_y1) > 1:
            return np.corrcoef(test_y1, pred1)[0, 1]
        else:
            return np.nan
    
    def MAE(self):
        assert self.test_y.shape == self.pred.shape, "矩阵维度不一致"
        return np.nanmean(np.abs(self.test_y - self.pred))
    
    def MAPE(self):
        valid_indices = ~np.isnan(self.test_y) & ~np.isnan(self.pred) & (self.test_y >= 0) & (self.pred >= 0)
        true_valid = self.test_y[valid_indices]
        pred_valid = self.pred[valid_indices]
        
        if len(true_valid) == 0:
            return np.nan
        
        epsilon = 1e-5
        true_valid = np.clip(true_valid, epsilon, None)
        pred_valid = np.clip(pred_valid, epsilon, None)
        
        mape = np.nanmean(np.abs((true_valid - pred_valid) / true_valid)) * 100  # 转换为百分比
        max_error = 500
        return min(mape, max_error)
    
    def main(self):
        r2 = self.R2_score()
        rmse = self.RMSE_score()
        r = self.correlation_coef()
        mae = self.MAE()
        mape = self.MAPE()
        
        print(f'\n{"="*60}')
        print(f'自定义指标计算结果（反归一化+掩码后）:')
        print(f'{"="*60}')
        print(f'Correlation coefficient (r): {r:.4f}')
        print(f'R2 Score:                    {r2:.4f}')
        print(f'RMSE:                        {rmse:.4f}')
        print(f'MAE:                         {mae:.4f}')
        print(f'MAPE:                        {mape:.4f}%')
        print(f'{"="*60}\n')
        
        return r, r2, rmse, mae, mape


class CuboidMovingMNISTPLModule(pl.LightningModule):

    def __init__(self,
                 total_num_steps: int,
                 oc_file: str = None,
                 save_dir: str = None,
                 loss_type: str = 'hybrid_scale_aware'):  # 新增参数：损失函数类型
        super(CuboidMovingMNISTPLModule, self).__init__()
        self._max_train_iter = total_num_steps
        if oc_file is not None:
            oc_from_file = OmegaConf.load(open(oc_file, "r"))
        else:
            oc_from_file = None
        oc = self.get_base_config(oc_from_file=oc_from_file)
        
        # 初始化损失函数
        self.loss_type = loss_type
        if loss_type == 'huber':
            self.criterion = HuberLossWithMask(delta=1.0)
            print("✓ 使用 Huber 损失函数（对极端值鲁棒）")
        elif loss_type == 'adaptive':
            # percentile_threshold: 高变异性样本的阈值
            # 90 = 降低最高10%样本的权重（推荐）
            # 95 = 降低最高5%样本的权重（更激进）
            self.criterion = AdaptiveLoss(base_loss='huber', percentile_threshold=90)
            print("✓ 使用自适应损失函数（降低高变异性样本权重）")
        elif loss_type == 'combined':
            self.criterion = CombinedLoss()
            print("✓ 使用组合损失函数（MSE + Huber + 加权）")
        elif loss_type == 'hybrid_scale_aware':
            # 🌟 新的混合尺度感知损失（推荐）
            self.criterion = HybridScaleAwareLoss(
                alpha=0.7,  # Huber Loss 权重，推荐 0.6-0.8
                delta=1.0,  # Huber Loss 阈值
                use_weighted_mse=False,  # 是否对低浓度区域加权
                weight_epsilon=1.0
            )
            print("✓ 使用混合尺度感知损失函数（Huber + MSE，在 log-z 空间）")
        elif loss_type == 'hybrid_weighted':
            # 带加权的版本（避免 MAPE 爆炸）
            self.criterion = HybridScaleAwareLoss(
                alpha=0.9,
                delta=2.0,
                use_weighted_mse=True,  # 对低浓度区域加权
                weight_epsilon=1.0
            )
            print("✓ 使用加权混合尺度感知损失函数（对低浓度区域额外关注）")
        elif loss_type == 'adaptive_hybrid':
            # 自适应混合损失（训练过程中动态调整）
            self.criterion = AdaptiveHybridLoss(
                alpha_start=0.5,  # 初始 alpha
                alpha_end=0.8,    # 最终 alpha
                warmup_epochs=50, # 预热轮数
                delta=1.0
            )
            print("✓ 使用自适应混合损失函数（训练过程中动态调整权重）")
        else:  # 'mse'
            self.criterion = nn.MSELoss()
            print("✓ 使用标准 MSE 损失函数")
        model_cfg = OmegaConf.to_object(oc.model)
        num_blocks = len(model_cfg["enc_depth"])
        if isinstance(model_cfg["self_pattern"], str):
            enc_attn_patterns = [model_cfg["self_pattern"]] * num_blocks
        else:
            enc_attn_patterns = OmegaConf.to_container(model_cfg["self_pattern"])
        if isinstance(model_cfg["cross_self_pattern"], str):
            dec_self_attn_patterns = [model_cfg["cross_self_pattern"]] * num_blocks
        else:
            dec_self_attn_patterns = OmegaConf.to_container(model_cfg["cross_self_pattern"])
        if isinstance(model_cfg["cross_pattern"], str):
            dec_cross_attn_patterns = [model_cfg["cross_pattern"]] * num_blocks
        else:
            dec_cross_attn_patterns = OmegaConf.to_container(model_cfg["cross_pattern"])

        self.torch_nn_module = CuboidTransformerModel(
            input_shape=model_cfg["input_shape"],
            target_shape=model_cfg["target_shape"],
            base_units=model_cfg["base_units"],
            block_units=model_cfg["block_units"],
            scale_alpha=model_cfg["scale_alpha"],
            enc_depth=model_cfg["enc_depth"],
            dec_depth=model_cfg["dec_depth"],
            enc_use_inter_ffn=model_cfg["enc_use_inter_ffn"],
            dec_use_inter_ffn=model_cfg["dec_use_inter_ffn"],
            dec_hierarchical_pos_embed=model_cfg["dec_hierarchical_pos_embed"],
            downsample=model_cfg["downsample"],
            downsample_type=model_cfg["downsample_type"],
            enc_attn_patterns=enc_attn_patterns,
            dec_self_attn_patterns=dec_self_attn_patterns,
            dec_cross_attn_patterns=dec_cross_attn_patterns,
            dec_cross_last_n_frames=model_cfg["dec_cross_last_n_frames"],
            dec_use_first_self_attn=model_cfg["dec_use_first_self_attn"],
            num_heads=model_cfg["num_heads"],
            attn_drop=model_cfg["attn_drop"],
            proj_drop=model_cfg["proj_drop"],
            ffn_drop=model_cfg["ffn_drop"],
            upsample_type=model_cfg["upsample_type"],
            ffn_activation=model_cfg["ffn_activation"],
            gated_ffn=model_cfg["gated_ffn"],
            norm_layer=model_cfg["norm_layer"],
            # global vectors
            num_global_vectors=model_cfg["num_global_vectors"],
            use_dec_self_global=model_cfg["use_dec_self_global"],
            dec_self_update_global=model_cfg["dec_self_update_global"],
            use_dec_cross_global=model_cfg["use_dec_cross_global"],
            use_global_vector_ffn=model_cfg["use_global_vector_ffn"],
            use_global_self_attn=model_cfg["use_global_self_attn"],
            separate_global_qkv=model_cfg["separate_global_qkv"],
            global_dim_ratio=model_cfg["global_dim_ratio"],
            # initial_downsample
            initial_downsample_type=model_cfg["initial_downsample_type"],
            initial_downsample_activation=model_cfg["initial_downsample_activation"],
            # initial_downsample_type=="conv"
            initial_downsample_scale=model_cfg["initial_downsample_scale"],
            initial_downsample_conv_layers=model_cfg["initial_downsample_conv_layers"],
            final_upsample_conv_layers=model_cfg["final_upsample_conv_layers"],
            # misc
            padding_type=model_cfg["padding_type"],
            z_init_method=model_cfg["z_init_method"],
            checkpoint_level=model_cfg["checkpoint_level"],
            pos_embed_type=model_cfg["pos_embed_type"],
            use_relative_pos=model_cfg["use_relative_pos"],
            self_attn_use_final_proj=model_cfg["self_attn_use_final_proj"],
            # residual blocks (NEW) - 从配置文件读取
            use_residual_blocks=model_cfg.get("use_residual_blocks", False),
            residual_block_num=model_cfg.get("residual_block_num", 2),
            residual_block_type=model_cfg.get("residual_block_type", "basic"),
            residual_block_position=model_cfg.get("residual_block_position", "encoder"),
            residual_use_3d=model_cfg.get("residual_use_3d", False),
            # initialization
            attn_linear_init_mode=model_cfg["attn_linear_init_mode"],
            ffn_linear_init_mode=model_cfg["ffn_linear_init_mode"],
            conv_init_mode=model_cfg["conv_init_mode"],
            down_up_linear_init_mode=model_cfg["down_up_linear_init_mode"],
            norm_init_mode=model_cfg["norm_init_mode"],
        )

        self.total_num_steps = total_num_steps
        if oc_file is not None:
            oc_from_file = OmegaConf.load(open(oc_file, "r"))
        else:
            oc_from_file = None
        oc = self.get_base_config(oc_from_file=oc_from_file)
        self.save_hyperparameters(oc)
        self.oc = oc
        # layout
        self.in_len = oc.layout.in_len
        self.out_len = oc.layout.out_len
        self.layout = oc.layout.layout
        # optimization
        self.max_epochs = oc.optim.max_epochs
        self.optim_method = oc.optim.method
        self.lr = oc.optim.lr
        self.wd = oc.optim.wd
        # lr_scheduler
        self.total_num_steps = total_num_steps
        self.lr_scheduler_mode = oc.optim.lr_scheduler_mode
        self.warmup_percentage = oc.optim.warmup_percentage
        self.min_lr_ratio = oc.optim.min_lr_ratio
        # logging
        self.save_dir = save_dir
        self.logging_prefix = oc.logging.logging_prefix
        # visualization
        self.train_example_data_idx_list = list(oc.vis.train_example_data_idx_list)
        self.val_example_data_idx_list = list(oc.vis.val_example_data_idx_list)
        self.test_example_data_idx_list = list(oc.vis.test_example_data_idx_list)
        self.eval_example_only = oc.vis.eval_example_only

        self.configure_save(cfg_file_path=oc_file)

        self.valid_mse = torchmetrics.MeanSquaredError()
        self.valid_mae = torchmetrics.MeanAbsoluteError()
        # 不使用 SSIM metric，改用函数式 API 避免显存累积
        # self.valid_ssim = torchmetrics.StructuralSimilarityIndexMeasure(data_range=1.0)

        self.test_mse = torchmetrics.MeanSquaredError()
        self.test_mae = torchmetrics.MeanAbsoluteError()
        # self.test_ssim = torchmetrics.StructuralSimilarityIndexMeasure()
        

    def configure_save(self, cfg_file_path=None):
        self.save_dir = os.path.join(exps_dir, self.save_dir)
        os.makedirs(self.save_dir, exist_ok=True)
        self.scores_dir = os.path.join(self.save_dir, 'scores')
        os.makedirs(self.scores_dir, exist_ok=True)
        if cfg_file_path is not None:
            cfg_file_target_path = os.path.join(self.save_dir, "cfg.yaml")
            if (not os.path.exists(cfg_file_target_path)) or \
                    (not os.path.samefile(cfg_file_path, cfg_file_target_path)):
                copyfile(cfg_file_path, cfg_file_target_path)
        self.example_save_dir = os.path.join(self.save_dir, "examples")
        os.makedirs(self.example_save_dir, exist_ok=True)

    def get_base_config(self, oc_from_file=None):
        oc = OmegaConf.create()
        oc.layout = self.get_layout_config()
        oc.optim = self.get_optim_config()
        oc.logging = self.get_logging_config()
        oc.trainer = self.get_trainer_config()
        oc.vis = self.get_vis_config()
        oc.model = self.get_model_config()
        if oc_from_file is not None:
            # oc = apply_omegaconf_overrides(oc, oc_from_file)
            oc = OmegaConf.merge(oc, oc_from_file)
        return oc

    @classmethod
    def get_model_config(cls):
        cfg = OmegaConf.create()
        height = 45
        width = 60
        in_len = 14
        out_len = 14
        in_channels = 6  # 输入通道数
        out_channels = 20  # 输出通道数
        cfg.input_shape = (in_len, height, width, in_channels)
        cfg.target_shape = (out_len, height, width, out_channels)

        # ⭐ 增加模型容量：base_units 从 64 增加到 128
        cfg.base_units = 128
        cfg.block_units = None # multiply by 2 when downsampling in each layer
        cfg.scale_alpha = 1.0

        # ⭐ 增加模型深度：每层从 1 个 block 增加到 2 个 block
        cfg.enc_depth = [2, 2]
        cfg.dec_depth = [2, 2]
        cfg.enc_use_inter_ffn = True
        cfg.dec_use_inter_ffn = True
        cfg.dec_hierarchical_pos_embed = True

        cfg.downsample = 2
        cfg.downsample_type = "patch_merge"
        cfg.upsample_type = "upsample"

        # ⭐ 增加全局向量数量：从 8 增加到 16
        cfg.num_global_vectors = 16
        cfg.use_dec_self_global = True
        cfg.dec_self_update_global = True
        cfg.use_dec_cross_global = True
        cfg.use_global_vector_ffn = True
        # ⭐ 启用全局自注意力
        cfg.use_global_self_attn = True
        cfg.separate_global_qkv = False
        cfg.global_dim_ratio = 1

        cfg.self_pattern = 'axial'
        cfg.cross_self_pattern = 'axial'
        cfg.cross_pattern = 'cross_1x1'
        cfg.dec_cross_last_n_frames = None

        cfg.attn_drop = 0.1
        cfg.proj_drop = 0.1
        cfg.ffn_drop = 0.1
        # ⭐ 增加注意力头数量：从 4 增加到 8
        cfg.num_heads = 8

        cfg.ffn_activation = 'gelu'
        cfg.gated_ffn = False
        cfg.norm_layer = 'layer_norm'
        cfg.padding_type = 'zeros'
        cfg.pos_embed_type = "t+hw"
        cfg.use_relative_pos = True
        cfg.self_attn_use_final_proj = True
        cfg.dec_use_first_self_attn = False

        cfg.z_init_method = 'zeros'  # The method for initializing the first input of the decoder
        cfg.initial_downsample_type = "conv"
        cfg.initial_downsample_activation = "leaky"
        cfg.initial_downsample_scale = 2
        cfg.initial_downsample_conv_layers = 2
        cfg.final_upsample_conv_layers = 1
        cfg.checkpoint_level = 2
        # initialization
        cfg.attn_linear_init_mode = "0"
        cfg.ffn_linear_init_mode = "0"
        cfg.conv_init_mode = "0"
        cfg.down_up_linear_init_mode = "0"
        cfg.norm_init_mode = "0"
        return cfg

    @staticmethod
    def get_layout_config():
        oc = OmegaConf.create()
        oc.in_len = 14
        oc.out_len = 14
        oc.layout = "NTHWC"  # The layout of the data, not the model
        return oc

    @staticmethod
    def get_optim_config():
        oc = OmegaConf.create()
        oc.seed = None
        oc.total_batch_size = 32
        oc.micro_batch_size = 8

        oc.method = "adamw"
        oc.lr = 1E-3
        oc.wd = 1E-5
        oc.gradient_clip_val = 1.0
        oc.max_epochs = 50
        # scheduler
        oc.warmup_percentage = 0.2
        oc.lr_scheduler_mode = "cosine"  # Can be strings like 'linear', 'cosine', 'platue'
        oc.min_lr_ratio = 0.1
        oc.warmup_min_lr_ratio = 0.1
        # early stopping
        oc.early_stop = False
        oc.early_stop_mode = "min"
        oc.early_stop_patience = 5
        oc.save_top_k = 1
        return oc

    @staticmethod
    def get_logging_config():
        oc = OmegaConf.create()
        oc.logging_prefix = "MovingMNIST"
        oc.monitor_lr = True
        oc.monitor_device = False
        oc.track_grad_norm = -1
        cfg.use_wandb = False
        return oc

    @staticmethod
    def get_trainer_config():
        oc = OmegaConf.create()
        oc.check_val_every_n_epoch = 1
        oc.log_step_ratio = 0.001  # Logging every 1% of the total training steps per epoch
        oc.precision = 32
        return oc

    @staticmethod
    def get_vis_config():
        oc = OmegaConf.create()
        oc.train_example_data_idx_list = [0, ]
        oc.val_example_data_idx_list = [0, ]
        oc.test_example_data_idx_list = [0, ]
        oc.eval_example_only = False
        return oc

    def configure_optimizers(self):
        # Configure the optimizer. Disable the weight decay for layer norm weights and all bias terms.
        decay_parameters = get_parameter_names(self.torch_nn_module, [nn.LayerNorm])
        decay_parameters = [name for name in decay_parameters if "bias" not in name]
        optimizer_grouped_parameters = [{
            'params': [p for n, p in self.torch_nn_module.named_parameters() if n in decay_parameters],
            'weight_decay': self.oc.optim.wd
        }, {
            'params': [p for n, p in self.torch_nn_module.named_parameters() if n not in decay_parameters],
            'weight_decay': 0.0
        }]

        if self.oc.optim.method == 'adamw':
            optimizer = torch.optim.AdamW(params=optimizer_grouped_parameters,
                                          lr=self.oc.optim.lr,
                                          weight_decay=self.oc.optim.wd)
        else:
            raise NotImplementedError

        warmup_iter = int(np.round(self.oc.optim.warmup_percentage * self.total_num_steps))

        if self.oc.optim.lr_scheduler_mode == 'cosine':
            warmup_scheduler = LambdaLR(optimizer,
                                        lr_lambda=warmup_lambda(warmup_steps=warmup_iter,
                                                                min_lr_ratio=self.oc.optim.warmup_min_lr_ratio))
            cosine_scheduler = CosineAnnealingLR(optimizer,
                                                 T_max=(self.total_num_steps - warmup_iter),
                                                 eta_min=self.oc.optim.min_lr_ratio * self.oc.optim.lr)
            lr_scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler],
                                        milestones=[warmup_iter])
            lr_scheduler_config = {
                'scheduler': lr_scheduler,
                'interval': 'step',
                'frequency': 1,
            }
        else:
            raise NotImplementedError
        return {'optimizer': optimizer, 'lr_scheduler': lr_scheduler_config}

    def set_trainer_kwargs(self, **kwargs):
        r"""
        Default kwargs used when initializing pl.Trainer
        """
        checkpoint_callback = ModelCheckpoint(
            monitor="valid_loss_step_epoch",  # 监控验证损失（与训练损失一致的计算方式）
            dirpath=os.path.join(self.save_dir, "checkpoints"),
            filename="model-{epoch:03d}",
            save_top_k=self.oc.optim.save_top_k,
            save_last=True,
            mode="min",
        )
        callbacks = kwargs.pop("callbacks", [])
        assert isinstance(callbacks, list)
        for ele in callbacks:
            assert isinstance(ele, Callback)
        callbacks += [checkpoint_callback, ]
        
        if self.oc.logging.monitor_lr:
            callbacks += [LearningRateMonitor(logging_interval='step'), ]
        if self.oc.logging.monitor_device:
            callbacks += [DeviceStatsMonitor(), ]
        if self.oc.optim.early_stop:
            callbacks += [EarlyStopping(monitor="valid_loss_step_epoch",  # 与checkpoint一致
                                        min_delta=0.0,
                                        patience=self.oc.optim.early_stop_patience,
                                        verbose=False,
                                        mode=self.oc.optim.early_stop_mode), ]
            
        logger = kwargs.pop("logger", [])
        tb_logger = pl_loggers.TensorBoardLogger(save_dir=self.save_dir)
        csv_logger = pl_loggers.CSVLogger(save_dir=self.save_dir)
        logger += [tb_logger, csv_logger]
        if self.oc.logging.use_wandb:
            wandb_logger = pl_loggers.WandbLogger(project=self.oc.logging.logging_prefix,
                                                  save_dir=self.save_dir)
            logger += [wandb_logger, ]

        log_every_n_steps = max(1, int(self.oc.trainer.log_step_ratio * self.total_num_steps))
        trainer_init_keys = inspect.signature(Trainer).parameters.keys()
        ret = dict(
            callbacks=callbacks,
            # log
            logger=logger,
            log_every_n_steps=log_every_n_steps,
            track_grad_norm=self.oc.logging.track_grad_norm,
            # save
            default_root_dir=self.save_dir,
            # ddp
            accelerator="gpu",
            strategy="ddp",
            #修改
            # strategy=ApexDDPStrategy(find_unused_parameters=False, delay_allreduce=True),
            
            # optimization
            max_epochs=self.oc.optim.max_epochs,
            check_val_every_n_epoch=self.oc.trainer.check_val_every_n_epoch,
            gradient_clip_val=self.oc.optim.gradient_clip_val,
            # NVIDIA amp
            precision=self.oc.trainer.precision,
            # 🔥 禁用 sanity check，避免初始极低值
            num_sanity_val_steps=0,
        )
        oc_trainer_kwargs = OmegaConf.to_object(self.oc.trainer)
        oc_trainer_kwargs = {key: val for key, val in oc_trainer_kwargs.items() if key in trainer_init_keys}
        ret.update(oc_trainer_kwargs)
        ret.update(kwargs)
        return ret

    @classmethod
    def get_total_num_steps(
            cls,
            num_samples: int,
            total_batch_size: int,
            epoch: int = None):
        r"""
        Parameters
        ----------
        num_samples:    int
            The number of samples of the datasets. `num_samples / micro_batch_size` is the number of steps per epoch.
        total_batch_size:   int
            `total_batch_size == micro_batch_size * world_size * grad_accum`
        """
        if epoch is None:
            epoch = cls.get_optim_config().max_epochs
        return int(epoch * num_samples / total_batch_size)

    @staticmethod
    def get_movingmnist_datamodule(micro_batch_size: int = 1):
        dm = MovingMNISTDataModule(batch_size=micro_batch_size)
        return dm

    def forward(self, batch):
        in_seq, target_seq = batch
        try:
            pred_seq = self.torch_nn_module(in_seq)
            # 使用配置的损失函数
            loss = self.criterion(pred_seq, target_seq)
            print(f"Loss ({self.loss_type}): {loss.item()}")
            return pred_seq, loss
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"[OOM ERROR] CUDA out of memory in forward pass!")
                print(f"[OOM ERROR] Input shape: {in_seq.shape}")
                print(f"[OOM ERROR] Target shape: {target_seq.shape}")
                # 清理缓存并重试
                torch.cuda.empty_cache()
                import gc
                gc.collect()
            raise e

    def training_step(self, batch, batch_idx):
        try:
            x, y = batch
            print(f"Input shape x: {x.shape}, Target shape y: {y.shape}")
            
            # 🔥 如果使用自适应混合损失，更新 epoch
            if isinstance(self.criterion, AdaptiveHybridLoss):
                self.criterion.set_epoch(self.current_epoch)
            
            y_hat, loss = self(batch)
            
            # 立即释放不需要的张量
            if batch_idx not in self.train_example_data_idx_list:
                # 如果不需要保存可视化，立即释放预测结果
                y_hat_detached = y_hat.detach()
            else:
                y_hat_detached = y_hat
                self.save_vis_step_end(
                    batch_idx=batch_idx,
                    in_seq=x, target_seq=y,
                    pred_seq=y_hat_detached,
                    mode="train"
                )
            
            self.log('train_loss', loss, on_step=True, on_epoch=False)
            
            # 定期清理显存碎片（每50步）
            if batch_idx % 50 == 0:
                torch.cuda.empty_cache()
            
            return loss
        except Exception as e:
            print(f"\n{'='*80}")
            print(f"[TRAINING ERROR] Epoch: {self.current_epoch}, Batch: {batch_idx}")
            print(f"[TRAINING ERROR] Error type: {type(e).__name__}")
            print(f"[TRAINING ERROR] Error message: {str(e)}")
            if hasattr(torch.cuda, 'memory_summary'):
                print(f"[TRAINING ERROR] GPU Memory Summary:")
                print(torch.cuda.memory_summary(device=None, abbreviated=True))
            print(f"{'='*80}\n")
            raise e

    def validation_step(self, batch, batch_idx):
        try:
            x, y = batch
            B, T_out, H, W, C = y.shape

            with torch.no_grad():
                # 计算预测和损失（使用相同的损失函数）
                y_hat, loss = self(batch)
                
                if self.precision == 16:
                    y_hat = y_hat.float()

                # 记录验证损失（与训练损失一致）
                self.log("valid_loss_step", loss, 
                        prog_bar=True, on_step=True, on_epoch=True)

                # ========== 计算额外的评估指标 ==========
                self.valid_mse.update(y_hat, y)
                self.valid_mae.update(y_hat, y)

                # ========== SSIM：直接计算，不累积 ==========
                y_hat_ = rearrange(y_hat, "b t h w c -> (b t) c h w")
                y_ = rearrange(y,     "b t h w c -> (b t) c h w")

                # 使用函数式 API，不累积到 buffer
                step_ssim = ssim_func(y_hat_, y_, data_range=1.0)

                self.log("valid_ssim_step", step_ssim,
                        prog_bar=True, on_step=True, on_epoch=False)
                
                # 立即删除临时变量
                del y_hat_, y_
                
                # ========== 只在指定的 batch 保存可视化 ==========
                if not self.eval_example_only or batch_idx in self.val_example_data_idx_list:
                    self.save_vis_step_end(
                        batch_idx=batch_idx,
                        in_seq=x, target_seq=y,
                        pred_seq=y_hat,
                        mode="val"
                    )

            return H, W
        except Exception as e:
            print(f"\n{'='*80}")
            print(f"[VALIDATION ERROR] Epoch: {self.current_epoch}, Batch: {batch_idx}")
            print(f"[VALIDATION ERROR] Error type: {type(e).__name__}")
            print(f"[VALIDATION ERROR] Error message: {str(e)}")
            if hasattr(torch.cuda, 'memory_summary'):
                print(f"[VALIDATION ERROR] GPU Memory Summary:")
                print(torch.cuda.memory_summary(device=None, abbreviated=True))
            print(f"{'='*80}\n")
            raise e


    def validation_epoch_end(self, outputs):
        # 🔥 跳过 sanity check 阶段的指标记录
        # PyTorch Lightning 的 sanity_checking 属性在 sanity check 期间为 True
        if self.trainer.sanity_checking:
            print(f"[SANITY CHECK] 跳过指标记录")
            self.valid_mse.reset()
            self.valid_mae.reset()
            return
        
        # torchmetrics 已经计算了平均值
        H, W = outputs[0]
        
        # 计算额外的评估指标（用于监控，不用于early stopping）
        frame_mse = self.valid_mse.compute() * H * W
        frame_mae = self.valid_mae.compute() * H * W

        self.log("valid_mse_epoch", frame_mse,
                prog_bar=True, on_step=False, on_epoch=True)
        self.log("valid_mae_epoch", frame_mae,
                prog_bar=True, on_step=False, on_epoch=True)
        
        # valid_loss_step 已经在 validation_step 中通过 on_epoch=True 自动聚合
        # 这里不需要再手动计算 valid_loss_epoch

        self.valid_mse.reset()
        self.valid_mae.reset()
        
        # 清理显存碎片
        torch.cuda.empty_cache()



    def test_step(self, batch, batch_idx):
        x, y = batch
        B, T_out, H, W, C = y.shape
        global result, target_result
        
        # 在第一个batch时重置result列表
        if batch_idx == 0:
            result = []
            target_result = []
            print(f"\n[测试开始] 重置result和target_result列表")
        
        # ========== 所有 batch 都进行预测和指标计算 ==========
        y_hat, loss = self(batch)  # 计算损失（使用相同的损失函数）
        
        if self.precision == 16:
            y_hat = y_hat.float()
            y = y.float()
        
        # 记录测试损失（与训练/验证一致）
        self.log('test_loss_step', loss, 
                prog_bar=True, on_step=True, on_epoch=True)
        
        # 计算额外的评估指标 - torchmetrics 会自动累积并计算平均值（归一化数据上的指标）
        self.test_mse.update(y_hat, y)
        self.test_mae.update(y_hat, y)
        
        # 使用函数式 API 计算 SSIM，避免显存累积
        y_hat_ssim = rearrange(y_hat, "b t h w c -> (b t) c h w")
        y_ssim = rearrange(y, "b t h w c -> (b t) c h w")
        step_ssim = ssim_func(y_hat_ssim, y_ssim, data_range=1.0)
        
        # 保存所有预测结果和真实值（用于后续反归一化和掩码处理）
        y_result = y_hat.detach().cpu()
        y_target = y.detach().cpu()
        result.append(np.array(y_result))
        target_result.append(np.array(y_target))
        
        if batch_idx % 100 == 0:  # 每 100 个 batch 打印一次，避免刷屏
            print(f"Batch {batch_idx}: 累计保存了 {len(result)} 个batch的预测和真实值")
        
        self.log('test_ssim_step', step_ssim, prog_bar=True, on_step=True, on_epoch=False)
        
        # ========== 只在指定的 batch 保存可视化 ==========
        if not self.eval_example_only or batch_idx in self.test_example_data_idx_list:
            self.save_vis_step_end(
                batch_idx=batch_idx,
                in_seq=x, target_seq=y,
                pred_seq=y_hat,
                mode="test"
            )
        
        return H, W

    def test_epoch_end(self, outputs):
        global result, target_result
        
        # 将 list 转换为 numpy array 并 reshape
        # result 是 list of arrays，每个 array 形状为 (batch_size, 20, 45, 60, 14)
        # 需要合并成 (total_samples, 20, 45, 60, 14)
        print(f"\n{'='*80}")
        print(f"[测试结果统计 - GPU {self.global_rank}]")
        print(f"原始 result list 长度: {len(result)} 个batch")
        if len(result) > 0:
            print(f"第一个 batch 形状: {result[0].shape}")
            print(f"最后一个 batch 形状: {result[-1].shape}")
        
        # 使用 concatenate 沿着 batch 维度合并当前GPU的结果
        result_array = np.concatenate(result, axis=0)
        target_array = np.concatenate(target_result, axis=0)
        print(f"当前GPU合并后的形状: pred={result_array.shape}, target={target_array.shape}")
        
        # ========== 收集所有GPU的结果（分批处理避免OOM）==========
        if self.trainer.world_size > 1:
            # 多GPU情况：分批收集所有GPU的结果以避免内存溢出
            # 计算分批大小（每批处理的样本数）
            batch_size = result_array.shape[0]
            chunk_size = max(1, batch_size // 4)  # 分成4批处理
            
            result_chunks = []
            target_chunks = []
            
            # 分批处理
            for i in range(0, batch_size, chunk_size):
                end_idx = min(i + chunk_size, batch_size)
                
                # 转换当前批次为tensor
                result_chunk = torch.from_numpy(result_array[i:end_idx]).to(self.device)
                target_chunk = torch.from_numpy(target_array[i:end_idx]).to(self.device)
                
                # 收集当前批次
                gathered_results = [torch.zeros_like(result_chunk) for _ in range(self.trainer.world_size)]
                gathered_targets = [torch.zeros_like(target_chunk) for _ in range(self.trainer.world_size)]
                
                torch.distributed.all_gather(gathered_results, result_chunk)
                torch.distributed.all_gather(gathered_targets, target_chunk)
                
                # 只在主进程保存
                if self.global_rank == 0:
                    result_chunks.append(torch.cat(gathered_results, dim=0).cpu())
                    target_chunks.append(torch.cat(gathered_targets, dim=0).cpu())
                
                # 清理GPU内存
                del result_chunk, target_chunk, gathered_results, gathered_targets
                torch.cuda.empty_cache()
            
            # 只在主进程（rank 0）保存完整结果并计算自定义指标
            if self.global_rank == 0:
                # 合并所有批次
                all_results = torch.cat(result_chunks, dim=0)
                all_targets = torch.cat(target_chunks, dim=0)
                result_array_full = all_results.numpy()
                target_array_full = all_targets.numpy()
                
                print(f"\n[所有GPU结果汇总]")
                print(f"GPU数量: {self.trainer.world_size}")
                print(f"每个GPU样本数: {result_array.shape[0]}")
                print(f"总样本数: {result_array_full.shape[0]}")
                print(f"预期样本数: 1448")
                
                if result_array_full.shape[0] != 1448:
                    print(f"⚠️  警告: 样本数不匹配! 差异: {1448 - result_array_full.shape[0]} 个样本")
                
                # 保存归一化的预测结果
                save_path = os.environ.get('EARTHFORMER_OCEAN_CHL_RESULTS_ROOT', 'results') + 'earthformer/result_31/result_31.npy'
                np.save(save_path, result_array_full)
                print(f"✓ 归一化预测结果已保存到: {save_path}")
                print(f"✓ 最终形状: {result_array_full.shape}")
                
                # ========== 反归一化和掩码处理 ==========
                print(f"\n{'='*80}")
                print(f"[开始反归一化和掩码处理]")
                print(f"{'='*80}")
                
                pred_denorm = output_inver(result_array_full)
                target_denorm = output_inver(target_array_full)
                
                # 加载掩码并应用
                output_mask_path = os.environ.get('EARTHFORMER_OCEAN_CHL_DATA_ROOT', 'data') + 'output_mask.npy'
                try:
                    output_mask = np.load(output_mask_path)
                    print(f"✓ 成功加载掩码文件: {output_mask_path}")
                    print(f"  掩码形状: {output_mask.shape}")
                    
                    pred_masked = set_mask(pred_denorm, output_mask)
                    target_masked = set_mask(target_denorm, output_mask)
                    
                    # 保存反归一化+掩码后的结果
                    save_path_denorm = os.environ.get('EARTHFORMER_OCEAN_CHL_RESULTS_ROOT', 'results') + 'earthformer/result_31/result_31_denorm_masked.npy'
                    save_path_target = os.environ.get('EARTHFORMER_OCEAN_CHL_RESULTS_ROOT', 'results') + 'earthformer/result_31/target_31_denorm_masked.npy'
                    np.save(save_path_denorm, pred_masked)
                    np.save(save_path_target, target_masked)
                    print(f"✓ 反归一化+掩码预测结果已保存到: {save_path_denorm}")
                    print(f"✓ 反归一化+掩码真实值已保存到: {save_path_target}")
                    
                    # ========== 计算自定义指标 ==========
                    print(f"\n{'='*80}")
                    print(f"[计算自定义指标]")
                    print(f"{'='*80}")
                    metrics = Metrics(target_masked, pred_masked)
                    r, r2, rmse, mae, mape = metrics.main()
                    
                    # 保存指标到文件
                    metrics_save_path = os.environ.get('EARTHFORMER_OCEAN_CHL_RESULTS_ROOT', 'results') + 'earthformer/result_31/custom_metrics.txt'
                    with open(metrics_save_path, 'w') as f:
                        f.write(f"自定义指标计算结果（反归一化+掩码后）:\n")
                        f.write(f"{'='*60}\n")
                        f.write(f"Correlation coefficient (r): {r:.4f}\n")
                        f.write(f"R2 Score:                    {r2:.4f}\n")
                        f.write(f"RMSE:                        {rmse:.4f}\n")
                        f.write(f"MAE:                         {mae:.4f}\n")
                        f.write(f"MAPE:                        {mape:.4f}%\n")
                        f.write(f"{'='*60}\n")
                    print(f"✓ 自定义指标已保存到: {metrics_save_path}")
                    
                except FileNotFoundError:
                    print(f"❌ 错误: 找不到掩码文件 {output_mask_path}")
                    print(f"   跳过掩码处理和自定义指标计算")
                except Exception as e:
                    print(f"❌ 处理掩码或计算指标时出错: {str(e)}")
                    import traceback
                    traceback.print_exc()
        else:
            # 单GPU情况：直接保存和处理
            print(f"单GPU模式，样本数: {result_array.shape[0]}")
            save_path = os.environ.get('EARTHFORMER_OCEAN_CHL_RESULTS_ROOT', 'results') + 'earthformer/result_31/result_31.npy'
            np.save(save_path, result_array)
            print(f"✓ 测试结果已保存到: {save_path}")
            print(f"✓ 最终形状: {result_array.shape}")
            
            # 单GPU也进行反归一化和掩码处理
            try:
                # 使用 log10 + RobustScaler 方法反归一化
                pred_denorm = output_inver(result_array)
                target_denorm = output_inver(target_array)
                
                output_mask_path = os.environ.get('EARTHFORMER_OCEAN_CHL_DATA_ROOT', 'data') + 'output_mask.npy'
                output_mask = np.load(output_mask_path)
                
                pred_masked = set_mask(pred_denorm, output_mask)
                target_masked = set_mask(target_denorm, output_mask)
                
                save_path_denorm = os.environ.get('EARTHFORMER_OCEAN_CHL_RESULTS_ROOT', 'results') + 'earthformer/result_31/result_31_denorm_masked.npy'
                save_path_target = os.environ.get('EARTHFORMER_OCEAN_CHL_RESULTS_ROOT', 'results') + 'earthformer/result_31/target_31_denorm_masked.npy'
                np.save(save_path_denorm, pred_masked)
                np.save(save_path_target, target_masked)
                
                metrics = Metrics(target_masked, pred_masked)
                metrics.main()
            except Exception as e:
                print(f"❌ 单GPU模式处理掩码或计算指标时出错: {str(e)}")
        
        print(f"{'='*80}\n")
        
        # 计算额外的评估指标（用于参考，不是主要损失）
        # torchmetrics 在DDP模式下会自动同步所有GPU的指标
        H, W = outputs[0]
        frame_mse = self.test_mse.compute() * H * W
        frame_mae = self.test_mae.compute() * H * W
        
        if self.global_rank == 0:
            print(f"\n{'='*80}")
            print(f"[PyTorch Lightning 指标 - 归一化数据上的结果]")
            print(f"{'='*80}")
            print(f"Test MSE (normalized):  {frame_mse:.6f}")
            print(f"Test MAE (normalized):  {frame_mae:.6f}")
            print(f"Test RMSE (normalized): {torch.sqrt(frame_mse):.6f}")
            print(f"{'='*80}\n")
            print(f"注意: 以上是归一化数据上的指标，仅供参考")
            print(f"      最终评估请使用上面的自定义指标（反归一化+掩码后）")
            print(f"{'='*80}\n")

        self.log('test_mse_epoch', frame_mse, prog_bar=True, on_step=False, on_epoch=True)
        self.log('test_mae_epoch', frame_mae, prog_bar=True, on_step=False, on_epoch=True)
        
        # test_loss_step_epoch 已经在 test_step 中通过 on_epoch=True 自动聚合
        
        self.test_mse.reset()
        self.test_mae.reset()

    def save_vis_step_end(
            self,
            batch_idx: int,
            in_seq: torch.Tensor, target_seq: torch.Tensor,
            pred_seq: torch.Tensor,
            mode: str = "train"):
        print(f"Saving visualization: mode={mode}, batch_idx={batch_idx}")
        print(f"pred_seq shape: {pred_seq.shape}, target_seq shape: {target_seq.shape}")

        if self.local_rank == 0:
            if mode == "train":
                example_data_idx_list = self.train_example_data_idx_list
            elif mode == "val":
                example_data_idx_list = self.val_example_data_idx_list
            elif mode == "test":
                example_data_idx_list = self.test_example_data_idx_list
            else:
                raise ValueError(f"Wrong mode {mode}! Must be in ['train', 'val', 'test'].")
            if batch_idx in example_data_idx_list:
                micro_batch_size = in_seq.shape[self.layout.find("N")]
                data_idx = int(batch_idx * micro_batch_size)
                # 立即转移到 CPU 并释放 GPU 引用
                in_seq_cpu = in_seq.detach().cpu().float().numpy()
                target_seq_cpu = target_seq.detach().cpu().float().numpy()
                pred_seq_cpu = pred_seq.detach().cpu().float().numpy()
                
                save_example_vis_results(
                    save_dir=self.example_save_dir,
                    save_prefix=f'{mode}_epoch_{self.current_epoch}_data_{data_idx}',
                    in_seq=in_seq_cpu,
                    target_seq=target_seq_cpu,
                    pred_seq=pred_seq_cpu,
                    layout=self.layout,
                    plot_stride=1,
                    label=self.oc.logging.logging_prefix)
                
                # 显式删除 CPU 副本
                del in_seq_cpu, target_seq_cpu, pred_seq_cpu

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save', default='tmp_mnist', type=str)
    parser.add_argument('--gpus', default=1, type=int)
    parser.add_argument('--cfg', default=None, type=str)
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--ckpt_name', default=None, type=str,
                        help='The model checkpoint trained on N-body MovingMNIST.')
    return parser

def main():
    parser = get_parser()
    args = parser.parse_args()
    if args.cfg is not None:
        oc_from_file = OmegaConf.load(open(args.cfg, "r"))
        total_batch_size = oc_from_file.optim.total_batch_size
        micro_batch_size = oc_from_file.optim.micro_batch_size
        max_epochs = oc_from_file.optim.max_epochs
        seed = oc_from_file.optim.seed
    else:
        micro_batch_size = 1
        total_batch_size = int(micro_batch_size * args.gpus)
        max_epochs = None
        seed = 0
    seed_everything(seed, workers=True)
    dm = CuboidMovingMNISTPLModule.get_movingmnist_datamodule(micro_batch_size=micro_batch_size)
    dm.prepare_data()
    dm.setup()
    # accumulate_grad_batches = total_batch_size // (micro_batch_size * args.gpus)
    #修改
    accumulate_grad_batches = max(1, total_batch_size // (micro_batch_size * args.gpus))
    total_num_steps = CuboidMovingMNISTPLModule.get_total_num_steps(
        epoch=max_epochs,
        num_samples=dm.num_train_samples,
        total_batch_size=total_batch_size,
    )
    pl_module = CuboidMovingMNISTPLModule(
        total_num_steps=total_num_steps,
        save_dir=args.save,
        oc_file=args.cfg,
        loss_type='hybrid_scale_aware')  # 🔥 使用Adaptive损失（测试结果最优，降低52.5%损失）
    trainer_kwargs = pl_module.set_trainer_kwargs(
        devices=args.gpus,
        accumulate_grad_batches=accumulate_grad_batches,
    )
    trainer = Trainer(**trainer_kwargs)
    
    # if args.test:
    #     assert args.ckpt_name is not None, f"args.ckpt_name is required for test!"
    #     ckpt_path = os.path.join(pl_module.save_dir, "checkpoints", args.ckpt_name)
    #     trainer.test(model=pl_module,
    #                  datamodule=dm,
    #                  ckpt_path=ckpt_path)
    # else:
    #     if args.ckpt_name is not None:
    #         ckpt_path = os.path.join(pl_module.save_dir, "checkpoints", args.ckpt_name)
    #         if not os.path.exists(ckpt_path):
    #             warnings.warn(f"ckpt {ckpt_path} not exists! Start training from epoch 0.")
    #             ckpt_path = None
    #     else:
    #         ckpt_path = None
    #     trainer.fit(model=pl_module,
    #                 datamodule=dm,
    #                 ckpt_path=ckpt_path)
    #     trainer.test(ckpt_path="best",
    #                  datamodule=dm)
    #修改如下
        # 5️⃣ 如果是测试模式
    # ===============================
    if args.test:
        assert args.ckpt_name is not None, f"args.ckpt_name is required for test!"
        ckpt_path = os.path.join(pl_module.save_dir, "checkpoints", args.ckpt_name)
        trainer.test(model=pl_module, datamodule=dm, ckpt_path=ckpt_path)
        return  # 测试完直接退出

    # ===============================
    # 6️⃣ 自动检测 checkpoint 并恢复训练
    # ===============================
    import warnings
    checkpoints_dir = os.path.join(pl_module.save_dir, "checkpoints")
    auto_last_ckpt = os.path.join(checkpoints_dir, "last.ckpt")

    if args.ckpt_name is not None:
        ckpt_path = os.path.join(checkpoints_dir, args.ckpt_name)
        if not os.path.exists(ckpt_path):
            warnings.warn(f"[WARN] 指定的 ckpt {ckpt_path} 不存在，将从头开始训练。")
            ckpt_path = None
        else:
            print(f"[INFO] 从指定的 ckpt 继续训练: {ckpt_path}")
    else:
        if os.path.exists(auto_last_ckpt):
            print(f"[INFO] 检测到上次训练的断点文件: {auto_last_ckpt}")
            print("[INFO] 将自动从上次中断的地方继续训练。")
            ckpt_path = auto_last_ckpt
        else:
            print("[INFO] 未检测到 checkpoint，将从头开始训练。")
            ckpt_path = None

    # ===============================
    # 7️⃣ 开始训练与测试
    # ===============================
    trainer.fit(model=pl_module, datamodule=dm, ckpt_path=ckpt_path)
    trainer.test(ckpt_path="best", datamodule=dm)


if __name__ == "__main__":
    main()
