"""
迁移学习训练脚本
从预训练模型（NEMO数据）迁移到 Argo 数据
# 1. 先创建日志目录
mkdir -p experiments/train_34

# 2. 运行训练并保存日志
CUDA_VISIBLE_DEVICES=0,1,2,3 \
python train_cuboid_mnist_transfer.py \
    --pretrain_ckpt experiments/train_29/checkpoints/last.ckpt \
    --cfg cfg_transfer_learning.yaml \
    --freeze_encoder \
    --freeze_decoder_layers 0 \
    --transfer_mask_path "$EARTHFORMER_OCEAN_CHL_DATA_ROOT/output_mask.npy" \
    --transfer_save_dir "$EARTHFORMER_OCEAN_CHL_RESULTS_ROOT/earthformer/result_34" \
    --gpus 4 \
    --save train_34 \
    2>&1 | tee -a experiments/train_34/train34.log

关键策略：
1. 加载预训练权重
2. 冻结编码器（可选）
3. 使用更小的学习率
4. 使用 moving_mnist_transfer 数据集
5. 完整保留预训练模型的所有功能：
   - 掩码处理
   - 反标准化
   - 自定义指标计算
   - 多GPU结果合并
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.dirname(__file__)+"/../../../src/"))
sys.path.append("../..")

import matplotlib
matplotlib.use('Agg')

import warnings
import argparse
import torch
import numpy as np
from pytorch_lightning import Trainer, seed_everything
from omegaconf import OmegaConf

# 导入预训练模块（包含所有功能）
import train_cuboid_mnist_29 as pretrain_module
from train_cuboid_mnist_29 import (
    CuboidMovingMNISTPLModule,
    output_inver,  # 反归一化函数
    set_mask,      # 掩码处理函数
    Metrics,       # 自定义指标计算类
)

# 导入迁移学习数据集
from earthformer_ocean_chl.data.argo_dataset import MovingMNISTDataModule as TransferDataModule

_curr_dir = os.path.realpath(os.path.dirname(os.path.realpath(__file__)))
exps_dir = os.path.join(_curr_dir, "experiments")


# ========== Argo 数据专用反归一化函数 ==========
def argo_output_inver(data, clip_range=None, eps=1e-6):
    """
    Argo 数据的 log10 + Z-score 反归一化
    
    ⚠️ 重要：这里的 mean 和 std 是 Argo 数据的统计量
    如果运行 check_argo_normalization.py 后发现需要更新，请修改下面的值
    
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
    
    # ===== ⭐ Argo 数据的 mean 和 std（可修改）=====
    # 🔧 如果运行 check_argo_normalization.py 后提示需要更新，请替换下面的值
    
    # 默认使用 NEMO 数据的统计量（如果 Argo 和 NEMO 分布相似）
    argo_mean_lst = [-0.8150518660569386, -0.8184554683231355, -0.822703524803779, -0.824855389808018, -0.8210954685828902, 
                     -0.8112116116265466, -0.7911747850388956, -0.766956232890517, -0.7538359693700075, -0.7596056466209652, 
                     -0.7846415772496859, -0.8260205611581808, -0.8802941139587199, -0.9491939325142774, -1.0310428008355457, 
                     -1.1171694803816496, -1.2901266069829311, -1.437523772764406, -1.7213580662831198, -1.9430977950423483]
    
    argo_std_lst = [0.38515552730214936, 0.3936268782678006, 0.40189568752311516, 0.40870158064834583, 0.40199455953901436, 
                    0.3765359072653078, 0.330584940704064, 0.2787614365694394, 0.25136622036734785, 0.2604293887160967, 
                    0.2929291813593174, 0.33331404749842125, 0.3590243068903036, 0.37619140416048674, 0.3849238792031455, 
                    0.3764520516549704, 0.3482062311070318, 0.3170262043432067, 0.2521287643262882, 0.21427301870926355]
    
    # 📝 如果需要使用 Argo 自己的统计量，请取消注释并填入数值：
    # argo_mean_lst = [
    #     -0.75, -0.78, ...  # 从 check_argo_normalization.py 的输出复制
    # ]
    # argo_std_lst = [
    #     0.42, 0.45, ...    # 从 check_argo_normalization.py 的输出复制
    # ]
    
    print(f"\n[Argo 数据 log10 + Z-score 反归一化]")
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
    if data.shape[-1] == len(argo_mean_lst):
        # (B, T, H, W, C) → (B, C, T, H, W)
        data = np.transpose(data, (0, 4, 1, 2, 3))
    
    B, C, T, H, W = data.shape
    if C != len(argo_mean_lst):
        print(f"⚠️ 通道数({C})与 mean/std 列表长度({len(argo_mean_lst)})不匹配")
        C = min(C, len(argo_mean_lst))
    
    # 转换为数组并调整形状
    mean_arr = np.array(argo_mean_lst[:C]).reshape(1, C, 1, 1, 1)
    std_arr = np.array(argo_std_lst[:C]).reshape(1, C, 1, 1, 1)
    
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


class TransferLearningModule(CuboidMovingMNISTPLModule):
    """
    迁移学习模块：继承预训练模块，添加权重冻结功能
    
    完整保留预训练模型的所有功能：
    - training_step, validation_step, test_step
    - validation_epoch_end, test_epoch_end
    - 反归一化 (output_inver)
    - 掩码处理 (set_mask)
    - 自定义指标计算 (Metrics)
    - 多GPU结果合并
    """
    
    def __init__(self, 
                 freeze_encoder=True,
                 freeze_decoder_layers=0,
                 transfer_output_mask_path=None,  # 新增：迁移学习的掩码路径
                 transfer_save_dir=None,          # 新增：迁移学习的结果保存路径
                 *args, **kwargs):
        """
        Args:
            freeze_encoder: 是否冻结编码器
            freeze_decoder_layers: 冻结解码器的前N层（0=不冻结）
            transfer_output_mask_path: Argo数据的掩码文件路径（如果为None，使用预训练的路径）
            transfer_save_dir: 迁移学习结果保存目录（如果为None，使用默认路径）
        """
        super().__init__(*args, **kwargs)
        self.freeze_encoder = freeze_encoder
        self.freeze_decoder_layers = freeze_decoder_layers
        
        # 迁移学习特定的路径配置
        self.transfer_output_mask_path = transfer_output_mask_path
        self.transfer_save_dir = transfer_save_dir
        
        # 如果没有指定迁移学习的保存路径，使用默认路径
        if self.transfer_save_dir is None:
            self.transfer_save_dir = os.environ.get('EARTHFORMER_OCEAN_CHL_RESULTS_ROOT', 'results') + 'earthformer/transfer_learning/'
        
        # 创建保存目录
        os.makedirs(self.transfer_save_dir, exist_ok=True)
        print(f"✓ 迁移学习结果将保存到: {self.transfer_save_dir}")
        
    def setup(self, stage):
        """在训练开始前冻结指定的层"""
        super().setup(stage)
        
        if stage == 'fit':
            self._freeze_layers()
    
    def _freeze_layers(self):
        """冻结指定的模型层"""
        frozen_params = 0
        total_params = 0
        
        for name, param in self.torch_nn_module.named_parameters():
            total_params += param.numel()
            
            # 冻结编码器
            if self.freeze_encoder and 'enc' in name:
                param.requires_grad = False
                frozen_params += param.numel()
                continue
            
            # 冻结解码器前N层
            if self.freeze_decoder_layers > 0 and 'dec' in name:
                # 检查是否在前N层
                for i in range(self.freeze_decoder_layers):
                    if f'dec_layers.{i}.' in name:
                        param.requires_grad = False
                        frozen_params += param.numel()
                        break
        
        print(f"\n{'='*60}")
        print(f"迁移学习权重冻结策略:")
        print(f"{'='*60}")
        print(f"冻结编码器: {self.freeze_encoder}")
        print(f"冻结解码器层数: {self.freeze_decoder_layers}")
        print(f"总参数量: {total_params:,}")
        print(f"冻结参数量: {frozen_params:,} ({100*frozen_params/total_params:.1f}%)")
        print(f"可训练参数量: {total_params-frozen_params:,} ({100*(total_params-frozen_params)/total_params:.1f}%)")
        print(f"{'='*60}\n")
    
    def test_step(self, batch, batch_idx):
        """
        重写 test_step，确保正确使用全局变量
        """
        x, y = batch
        B, T_out, H, W, C = y.shape
        
        # 在第一个batch时重置result列表
        if batch_idx == 0:
            pretrain_module.result = []
            pretrain_module.target_result = []
            print(f"\n[迁移学习测试开始] 重置result和target_result列表")
        
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
        from einops import rearrange
        from torchmetrics.functional import structural_similarity_index_measure as ssim_func
        y_hat_ssim = rearrange(y_hat, "b t h w c -> (b t) c h w")
        y_ssim = rearrange(y, "b t h w c -> (b t) c h w")
        step_ssim = ssim_func(y_hat_ssim, y_ssim, data_range=1.0)
        
        # 保存所有预测结果和真实值（用于后续反归一化和掩码处理）
        y_result = y_hat.detach().cpu()
        y_target = y.detach().cpu()
        pretrain_module.result.append(np.array(y_result))
        pretrain_module.target_result.append(np.array(y_target))
        
        if batch_idx % 100 == 0:  # 每 100 个 batch 打印一次，避免刷屏
            print(f"Batch {batch_idx}: 累计保存了 {len(pretrain_module.result)} 个batch的预测和真实值")
        
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
        """
        重写test_epoch_end，使用迁移学习的路径配置
        完整保留预训练模型的所有功能
        """
        # 使用模块级别的全局变量
        result = pretrain_module.result
        target_result = pretrain_module.target_result
        
        # 将 list 转换为 numpy array 并 reshape
        print(f"\n{'='*80}")
        print(f"[迁移学习测试结果统计 - GPU {self.global_rank}]")
        print(f"原始 result list 长度: {len(result)} 个batch")
        
        # 检查 result 是否为空
        if len(result) == 0:
            print(f"\n{'='*80}")
            print(f"❌ 错误：result 列表为空！")
            print(f"可能原因：")
            print(f"  1. test_step 没有正确保存预测结果")
            print(f"  2. 全局变量访问有问题")
            print(f"  3. 数据集为空或没有数据")
            print(f"{'='*80}\n")
            return
        
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
                
                # 保存归一化的预测结果
                save_path = os.path.join(self.transfer_save_dir, 'result_normalized.npy')
                np.save(save_path, result_array_full)
                print(f"✓ 归一化预测结果已保存到: {save_path}")
                print(f"✓ 最终形状: {result_array_full.shape}")
                
                # ========== 反归一化和掩码处理 ==========
                print(f"\n{'='*80}")
                print(f"[开始反归一化和掩码处理 - 使用 Argo 数据统计量]")
                print(f"{'='*80}")
                
                # 先反归一化（使用 Argo 数据的统计量）
                pred_denorm = argo_output_inver(result_array_full)
                target_denorm = argo_output_inver(target_array_full)
                
                # 再应用掩码
                # 使用迁移学习的掩码路径（如果指定），否则使用预训练的路径
                if self.transfer_output_mask_path is not None:
                    output_mask_path = self.transfer_output_mask_path
                else:
                    output_mask_path = os.environ.get('EARTHFORMER_OCEAN_CHL_DATA_ROOT', 'data') + 'output_mask.npy'
                
                try:
                    output_mask = np.load(output_mask_path)
                    print(f"✓ 成功加载掩码文件: {output_mask_path}")
                    print(f"  掩码形状: {output_mask.shape}")
                    
                    pred_masked = set_mask(pred_denorm, output_mask)
                    target_masked = set_mask(target_denorm, output_mask)
                    
                    # 保存反归一化+掩码后的结果
                    save_path_denorm = os.path.join(self.transfer_save_dir, 'result_denorm_masked.npy')
                    save_path_target = os.path.join(self.transfer_save_dir, 'target_denorm_masked.npy')
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
                    metrics_save_path = os.path.join(self.transfer_save_dir, 'custom_metrics.txt')
                    with open(metrics_save_path, 'w') as f:
                        f.write(f"迁移学习自定义指标计算结果（反归一化+掩码后）:\n")
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
            save_path = os.path.join(self.transfer_save_dir, 'result_normalized.npy')
            np.save(save_path, result_array)
            print(f"✓ 归一化预测结果已保存到: {save_path}")
            
            # 单GPU也进行反归一化和掩码处理
            print(f"\n{'='*80}")
            print(f"[开始反归一化和掩码处理 - 使用 Argo 数据统计量]")
            print(f"{'='*80}")
            
            pred_denorm = argo_output_inver(result_array)
            target_denorm = argo_output_inver(target_array)
            
            # 使用迁移学习的掩码路径
            if self.transfer_output_mask_path is not None:
                output_mask_path = self.transfer_output_mask_path
            else:
                output_mask_path = os.environ.get('EARTHFORMER_OCEAN_CHL_DATA_ROOT', 'data') + 'output_mask.npy'
            
            try:
                output_mask = np.load(output_mask_path)
                print(f"✓ 成功加载掩码文件: {output_mask_path}")
                
                pred_masked = set_mask(pred_denorm, output_mask)
                target_masked = set_mask(target_denorm, output_mask)
                
                # 保存结果
                save_path_denorm = os.path.join(self.transfer_save_dir, 'result_denorm_masked.npy')
                save_path_target = os.path.join(self.transfer_save_dir, 'target_denorm_masked.npy')
                np.save(save_path_denorm, pred_masked)
                np.save(save_path_target, target_masked)
                print(f"✓ 反归一化+掩码结果已保存")
                
                # 计算自定义指标
                metrics = Metrics(target_masked, pred_masked)
                r, r2, rmse, mae, mape = metrics.main()
                
                # 保存指标
                metrics_save_path = os.path.join(self.transfer_save_dir, 'custom_metrics.txt')
                with open(metrics_save_path, 'w') as f:
                    f.write(f"迁移学习自定义指标计算结果（反归一化+掩码后）:\n")
                    f.write(f"{'='*60}\n")
                    f.write(f"Correlation coefficient (r): {r:.4f}\n")
                    f.write(f"R2 Score:                    {r2:.4f}\n")
                    f.write(f"RMSE:                        {rmse:.4f}\n")
                    f.write(f"MAE:                         {mae:.4f}\n")
                    f.write(f"MAPE:                        {mape:.4f}%\n")
                    f.write(f"{'='*60}\n")
                print(f"✓ 自定义指标已保存到: {metrics_save_path}")
                
            except Exception as e:
                print(f"❌ 处理掩码或计算指标时出错: {str(e)}")
                import traceback
                traceback.print_exc()


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save', default='transfer_learning_exp', type=str,
                        help='实验保存目录名')
    parser.add_argument('--gpus', default=1, type=int,
                        help='GPU数量')
    parser.add_argument('--cfg', default=None, type=str,
                        help='配置文件路径')
    parser.add_argument('--test', action='store_true',
                        help='是否只进行测试')
    parser.add_argument('--ckpt_name', default=None, type=str,
                        help='checkpoint文件名（用于恢复训练或测试）')
    
    # 迁移学习特有参数
    parser.add_argument('--pretrain_ckpt', required=True, type=str,
                        help='预训练模型checkpoint路径（必需）')
    parser.add_argument('--freeze_encoder', action='store_true', default=True,
                        help='是否冻结编码器（默认True）')
    parser.add_argument('--freeze_decoder_layers', default=0, type=int,
                        help='冻结解码器的前N层（默认0=不冻结）')
    parser.add_argument('--transfer_mask_path', default=None, type=str,
                        help='Argo数据的掩码文件路径（可选，默认使用预训练的掩码）')
    parser.add_argument('--transfer_save_dir', default=None, type=str,
                        help='迁移学习结果保存目录（可选，默认使用自动生成的路径）')
    
    return parser


def load_pretrained_weights(pl_module, pretrain_ckpt_path):
    """
    加载预训练权重
    
    Args:
        pl_module: TransferLearningModule 实例
        pretrain_ckpt_path: 预训练checkpoint路径
    """
    print(f"\n{'='*60}")
    print(f"加载预训练权重:")
    print(f"{'='*60}")
    print(f"预训练模型路径: {pretrain_ckpt_path}")
    
    if not os.path.exists(pretrain_ckpt_path):
        raise FileNotFoundError(f"预训练模型不存在: {pretrain_ckpt_path}")
    
    # 加载checkpoint
    checkpoint = torch.load(pretrain_ckpt_path, map_location='cpu')
    
    # 提取模型权重
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    
    # 加载权重（strict=False 允许部分加载）
    missing_keys, unexpected_keys = pl_module.load_state_dict(state_dict, strict=False)
    
    print(f"✓ 预训练权重加载完成")
    if missing_keys:
        print(f"⚠️ 缺失的键 ({len(missing_keys)}): {missing_keys[:5]}...")
    if unexpected_keys:
        print(f"⚠️ 未预期的键 ({len(unexpected_keys)}): {unexpected_keys[:5]}...")
    print(f"{'='*60}\n")
    
    return pl_module


def main():
    parser = get_parser()
    args = parser.parse_args()
    
    # 加载配置文件
    if args.cfg is not None:
        oc_from_file = OmegaConf.load(open(args.cfg, "r"))
        total_batch_size = oc_from_file.optim.total_batch_size
        micro_batch_size = oc_from_file.optim.micro_batch_size
        max_epochs = oc_from_file.optim.max_epochs
        seed = oc_from_file.optim.seed
    else:
        micro_batch_size = 8
        total_batch_size = int(micro_batch_size * args.gpus)
        max_epochs = 100
        seed = 0
    
    seed_everything(seed, workers=True)
    
    # ========== 使用迁移学习数据集 ==========
    print(f"\n{'='*60}")
    print(f"初始化迁移学习数据集 (Argo数据)")
    print(f"{'='*60}")
    dm = TransferDataModule(
        batch_size=micro_batch_size,
        data_range=(0, 2909, 1),  # 根据你的实际数据范围调整
        train_ratio=0.7,
        val_ratio=0.1,
        test_ratio=0.2
    )
    dm.prepare_data()
    dm.setup()
    print(f"✓ 数据集初始化完成")
    print(f"  训练集: {dm.num_train_samples} 样本")
    print(f"  验证集: {dm.num_val_samples} 样本")
    print(f"  测试集: {dm.num_test_samples} 样本")
    print(f"{'='*60}\n")
    
    # 计算训练步数
    accumulate_grad_batches = max(1, total_batch_size // (micro_batch_size * args.gpus))
    total_num_steps = TransferLearningModule.get_total_num_steps(
        epoch=max_epochs,
        num_samples=dm.num_train_samples,
        total_batch_size=total_batch_size,
    )
    
    # ========== 创建迁移学习模块 ==========
    pl_module = TransferLearningModule(
        total_num_steps=total_num_steps,
        save_dir=args.save,
        oc_file=args.cfg,
        loss_type='hybrid_scale_aware',  # 使用相同的损失函数
        freeze_encoder=args.freeze_encoder,
        freeze_decoder_layers=args.freeze_decoder_layers,
        transfer_output_mask_path=args.transfer_mask_path,
        transfer_save_dir=args.transfer_save_dir
    )
    
    # ========== 加载预训练权重 ==========
    pl_module = load_pretrained_weights(pl_module, args.pretrain_ckpt)
    
    # ========== 配置Trainer ==========
    trainer_kwargs = pl_module.set_trainer_kwargs(
        devices=args.gpus,
        accumulate_grad_batches=accumulate_grad_batches,
    )
    trainer = Trainer(**trainer_kwargs)
    
    # ========== 训练或测试 ==========
    if args.test:
        assert args.ckpt_name is not None, "测试模式需要指定 --ckpt_name"
        ckpt_path = os.path.join(pl_module.save_dir, "checkpoints", args.ckpt_name)
        trainer.test(model=pl_module, datamodule=dm, ckpt_path=ckpt_path)
        return
    
    # 自动检测checkpoint
    checkpoints_dir = os.path.join(pl_module.save_dir, "checkpoints")
    auto_last_ckpt = os.path.join(checkpoints_dir, "last.ckpt")
    
    if args.ckpt_name is not None:
        ckpt_path = os.path.join(checkpoints_dir, args.ckpt_name)
        if not os.path.exists(ckpt_path):
            warnings.warn(f"指定的 ckpt {ckpt_path} 不存在，将从预训练权重开始训练。")
            ckpt_path = None
        else:
            print(f"从指定的 ckpt 继续训练: {ckpt_path}")
    else:
        if os.path.exists(auto_last_ckpt):
            print(f"检测到上次训练的断点: {auto_last_ckpt}")
            ckpt_path = auto_last_ckpt
        else:
            print("从预训练权重开始迁移学习训练")
            ckpt_path = None
    
    # 开始训练
    trainer.fit(model=pl_module, datamodule=dm, ckpt_path=ckpt_path)
    trainer.test(ckpt_path="best", datamodule=dm)


if __name__ == "__main__":
    main()
