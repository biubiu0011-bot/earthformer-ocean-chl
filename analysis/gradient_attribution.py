"""
梯度归因分析：使用积分梯度等方法分析输入变量对预测结果的贡献
"""
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from captum.attr import IntegratedGradients, GradientShap, DeepLift
from captum.attr import visualization as viz
import os

class GradientAttributionAnalyzer:
    def __init__(self, model, device='cuda'):
        self.model = model
        self.device = device
        self.model.eval()
        
        # 海表变量名称
        self.surface_vars = ['chla', 'sst', 'ssh', 'sss', 'ssu', 'ssv']
        
    def prepare_model_for_attribution(self):
        """为归因分析准备模型包装器"""
        def model_wrapper(x):
            # 输入格式：(batch_size, time, height, width, channels)
            # 需要构造完整的batch格式
            dummy_target = torch.zeros(x.shape[0], 14, 45, 60, 20).to(self.device)
            batch = (x, dummy_target)
            
            # 只返回预测结果，不返回损失
            with torch.no_grad():
                pred, _ = self.model(batch)
            return pred
        
        return model_wrapper
    
    def analyze_integrated_gradients(self, input_batch, target_depth_idx=0, save_dir='gradient_analysis'):
        """使用积分梯度分析输入变量的重要性"""
        os.makedirs(save_dir, exist_ok=True)
        
        x, y = input_batch
        x = x.to(self.device).requires_grad_(True)
        
        # 准备模型包装器
        model_wrapper = self.prepare_model_for_attribution()
        
        # 初始化积分梯度
        ig = IntegratedGradients(model_wrapper)
        
        # 创建基线（全零输入）
        baseline = torch.zeros_like(x)
        
        # 计算积分梯度
        print(f"计算积分梯度...")
        attributions = ig.attribute(
            x, 
            baseline, 
            target=target_depth_idx,  # 分析特定深度层
            n_steps=50
        )
        
        # 分析结果
        self._analyze_attribution_results(attributions, x, target_depth_idx, save_dir)
        
        return attributions
    
    def analyze_gradient_shap(self, input_batch, background_batch, target_depth_idx=0, save_dir='gradient_analysis'):
        """使用GradientSHAP分析"""
        x, y = input_batch
        x_bg, _ = background_batch
        
        x = x.to(self.device).requires_grad_(True)
        x_bg = x_bg.to(self.device)
        
        model_wrapper = self.prepare_model_for_attribution()
        
        # 初始化GradientSHAP
        gs = GradientShap(model_wrapper)
        
        print(f"计算GradientSHAP...")
        attributions = gs.attribute(
            x,
            baselines=x_bg,
            target=target_depth_idx,
            n_samples=50
        )
        
        self._analyze_attribution_results(attributions, x, target_depth_idx, save_dir, method='GradientSHAP')
        
        return attributions
    
    def _analyze_attribution_results(self, attributions, inputs, target_depth_idx, save_dir, method='IntegratedGradients'):
        """分析归因结果"""
        # 转换为numpy
        attr_np = attributions.detach().cpu().numpy()
        input_np = inputs.detach().cpu().numpy()
        
        # 计算每个变量的总体重要性
        var_importance = self._calculate_variable_importance(attr_np)
        
        # 绘制结果
        self._plot_attribution_results(attr_np, input_np, var_importance, target_depth_idx, save_dir, method)
        
        # 保存数值结果
        np.save(f'{save_dir}/{method}_attributions_depth_{target_depth_idx}.npy', attr_np)
        np.save(f'{save_dir}/{method}_variable_importance_depth_{target_depth_idx}.npy', var_importance)
        
        return var_importance
    
    def _calculate_variable_importance(self, attributions):
        """计算每个变量的重要性得分"""
        # attributions shape: (batch, time, height, width, channels)
        B, T, H, W, C = attributions.shape
        
        # 计算每个通道（变量）的平均绝对归因值
        var_importance = {}
        for c, var_name in enumerate(self.surface_vars):
            if c < C:
                # 计算该变量在所有时空位置上的平均绝对归因
                importance = np.mean(np.abs(attributions[:, :, :, :, c]))
                var_importance[var_name] = importance
        
        return var_importance
    
    def _plot_attribution_results(self, attributions, inputs, var_importance, target_depth_idx, save_dir, method):
        """绘制归因分析结果"""
        
        # 1. 变量重要性柱状图
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 重要性排序
        sorted_vars = sorted(var_importance.items(), key=lambda x: x[1], reverse=True)
        var_names = [item[0] for item in sorted_vars]
        var_scores = [item[1] for item in sorted_vars]
        
        bars = axes[0, 0].bar(var_names, var_scores, color='lightcoral', edgecolor='darkred')
        axes[0, 0].set_title(f'{method} - Variable Importance (Depth {target_depth_idx})')
        axes[0, 0].set_ylabel('Attribution Score')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # 添加数值标签
        for bar, score in zip(bars, var_scores):
            axes[0, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(var_scores)*0.01,
                           f'{score:.4f}', ha='center', va='bottom', fontsize=10)
        
        # 2. 时间序列重要性
        time_importance = np.mean(np.abs(attributions), axis=(0, 2, 3, 4))  # 平均over batch, H, W, C
        axes[0, 1].plot(range(len(time_importance)), time_importance, 'o-', color='blue', linewidth=2)
        axes[0, 1].set_title('Temporal Importance Pattern')
        axes[0, 1].set_xlabel('Time Step')
        axes[0, 1].set_ylabel('Average Attribution')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. 空间重要性热力图（选择第一个时间步和第一个变量）
        if attributions.shape[0] > 0:
            spatial_attr = attributions[0, 0, :, :, 0]  # 第一个样本，第一个时间步，第一个变量
            im1 = axes[1, 0].imshow(spatial_attr, cmap='RdBu_r', aspect='auto')
            axes[1, 0].set_title(f'Spatial Attribution Pattern ({self.surface_vars[0]})')
            axes[1, 0].set_xlabel('Width')
            axes[1, 0].set_ylabel('Height')
            plt.colorbar(im1, ax=axes[1, 0])
        
        # 4. 各变量的时间重要性对比
        for c, var_name in enumerate(self.surface_vars[:min(6, attributions.shape[-1])]):
            var_time_importance = np.mean(np.abs(attributions[:, :, :, :, c]), axis=(0, 2, 3))
            axes[1, 1].plot(range(len(var_time_importance)), var_time_importance, 
                           'o-', label=var_name, linewidth=2, markersize=4)
        
        axes[1, 1].set_title('Variable Importance Over Time')
        axes[1, 1].set_xlabel('Time Step')
        axes[1, 1].set_ylabel('Attribution Score')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{save_dir}/{method}_analysis_depth_{target_depth_idx}.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def analyze_multiple_depths(self, dataloader, depths_to_analyze=[0, 5, 10, 15, 19], save_dir='gradient_analysis'):
        """分析多个深度层的归因模式"""
        os.makedirs(save_dir, exist_ok=True)
        
        # 获取一个batch用于分析
        sample_batch = next(iter(dataloader))
        
        all_importance = {}
        
        for depth_idx in depths_to_analyze:
            print(f"\n分析深度层 {depth_idx}...")
            
            # 积分梯度分析
            var_importance = self.analyze_integrated_gradients(
                sample_batch, 
                target_depth_idx=depth_idx, 
                save_dir=save_dir
            )
            
            all_importance[f'depth_{depth_idx}'] = var_importance
        
        # 绘制深度对比图
        self._plot_depth_comparison(all_importance, depths_to_analyze, save_dir)
        
        return all_importance
    
    def _plot_depth_comparison(self, all_importance, depths_to_analyze, save_dir):
        """绘制不同深度的变量重要性对比"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # 1. 热力图：变量 x 深度
        importance_matrix = []
        for depth_idx in depths_to_analyze:
            depth_key = f'depth_{depth_idx}'
            if depth_key in all_importance:
                row = [all_importance[depth_key].get(var, 0) for var in self.surface_vars]
                importance_matrix.append(row)
        
        importance_matrix = np.array(importance_matrix)
        
        im = ax1.imshow(importance_matrix.T, cmap='viridis', aspect='auto')
        ax1.set_xticks(range(len(depths_to_analyze)))
        ax1.set_xticklabels([f'D{d}' for d in depths_to_analyze])
        ax1.set_yticks(range(len(self.surface_vars)))
        ax1.set_yticklabels(self.surface_vars)
        ax1.set_xlabel('Depth Layer')
        ax1.set_ylabel('Surface Variables')
        ax1.set_title('Variable Importance Across Depths')
        
        # 添加数值标签
        for i in range(len(depths_to_analyze)):
            for j in range(len(self.surface_vars)):
                text = ax1.text(i, j, f'{importance_matrix[i, j]:.3f}',
                               ha="center", va="center", color="white", fontsize=8)
        
        plt.colorbar(im, ax=ax1)
        
        # 2. 线图：每个变量随深度的变化
        for j, var_name in enumerate(self.surface_vars):
            var_scores = importance_matrix[:, j]
            ax2.plot(depths_to_analyze, var_scores, 'o-', label=var_name, linewidth=2, markersize=6)
        
        ax2.set_xlabel('Depth Layer')
        ax2.set_ylabel('Importance Score')
        ax2.set_title('Variable Importance Trends with Depth')
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{save_dir}/depth_comparison_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()