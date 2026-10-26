"""
快速分析脚本 - 简化版本，用于测试和快速分析
只运行敏感性分析，避免复杂的依赖问题
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path
import argparse

# 简化的敏感性分析器
class QuickSensitivityAnalyzer:
    def __init__(self, model, device='cuda'):
        self.model = model
        self.device = device
        self.model.eval()
        
        # 海表变量名称
        self.surface_vars = ['chla', 'sst', 'sss', 'ssh', 'ssu', 'ssv']
        
    def quick_perturbation_test(self, sample_batch, save_dir='quick_analysis'):
        """快速扰动测试"""
        os.makedirs(save_dir, exist_ok=True)
        
        x, y = sample_batch
        x = x.to(self.device)
        y = y.to(self.device)
        
        print(f"输入数据形状: {x.shape}")
        print(f"目标数据形状: {y.shape}")
        
        # 基线预测
        with torch.no_grad():
            baseline_pred, _ = self.model((x, y))
            baseline_mse = torch.nn.functional.mse_loss(baseline_pred, y).item()
        
        print(f"基线MSE: {baseline_mse:.6f}")
        
        # 扰动分析
        perturbation_level = 0.5  # 固定扰动级别
        results = {}
        
        for var_idx, var_name in enumerate(self.surface_vars):
            if var_idx >= x.shape[-1]:
                continue
                
            # 添加扰动
            x_perturbed = x.clone()
            noise_std = perturbation_level * torch.std(x[..., var_idx])
            noise = torch.randn_like(x_perturbed[..., var_idx]) * noise_std
            x_perturbed[..., var_idx] += noise
            
            # 预测
            with torch.no_grad():
                perturbed_pred, _ = self.model((x_perturbed, y))
                perturbed_mse = torch.nn.functional.mse_loss(perturbed_pred, y).item()
            
            mse_change = (perturbed_mse - baseline_mse) / baseline_mse * 100
            results[var_name] = mse_change
            
            print(f"{var_name}: {mse_change:+.2f}% MSE变化")
        
        # 可视化结果
        self._plot_quick_results(results, save_dir)
        
        return results
    
    def _plot_quick_results(self, results, save_dir):
        """绘制快速分析结果"""
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        var_names = list(results.keys())
        mse_changes = list(results.values())
        
        # 按重要性排序
        sorted_items = sorted(zip(var_names, mse_changes), key=lambda x: abs(x[1]), reverse=True)
        var_names_sorted = [item[0] for item in sorted_items]
        mse_changes_sorted = [item[1] for item in sorted_items]
        
        bars = ax.bar(var_names_sorted, mse_changes_sorted, 
                     color=['red' if x > 0 else 'blue' for x in mse_changes_sorted],
                     alpha=0.7, edgecolor='black')
        
        ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax.set_title('海表变量敏感性分析 (快速测试)', fontsize=14, fontweight='bold')
        ax.set_ylabel('MSE变化 (%)')
        ax.set_xlabel('海表变量')
        ax.tick_params(axis='x', rotation=45)
        
        # 添加数值标签
        for bar, change in zip(bars, mse_changes_sorted):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., 
                   height + (max(mse_changes_sorted) - min(mse_changes_sorted))*0.01,
                   f'{change:+.1f}%', ha='center', 
                   va='bottom' if height >= 0 else 'top', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f'{save_dir}/quick_sensitivity_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"结果图表已保存到: {save_dir}/quick_sensitivity_analysis.png")


def load_model_simple(model_path):
    """简单的模型加载函数"""
    print(f"加载模型: {model_path}")
    
    # 直接加载checkpoint
    checkpoint = torch.load(model_path, map_location='cpu')
    
    # 这里需要根据你的具体模型结构来调整
    # 如果有具体的模型类，可以这样加载：
    # model = YourModelClass.load_from_checkpoint(model_path)
    
    print("模型加载完成")
    return checkpoint  # 临时返回checkpoint，实际使用时需要返回模型实例


def create_dummy_data():
    """创建虚拟数据用于测试"""
    # 创建符合模型输入格式的虚拟数据
    batch_size = 2
    time_steps = 14
    height = 45
    width = 60
    channels = 6  # 6个海表变量
    
    x = torch.randn(batch_size, time_steps, height, width, channels)
    y = torch.randn(batch_size, 14, height, width, 20)  # 20个深度层
    
    return (x, y)


def main():
    parser = argparse.ArgumentParser(description='快速敏感性分析')
    parser.add_argument('--model_path', type=str, 
                       default='experiments/train_34/checkpoints/best.ckpt',
                       help='模型checkpoint路径')
    parser.add_argument('--save_dir', type=str, default='quick_analysis',
                       help='结果保存目录')
    parser.add_argument('--use_dummy_data', action='store_true',
                       help='使用虚拟数据进行测试')
    
    args = parser.parse_args()
    
    print("=" * 50)
    print("快速敏感性分析")
    print("=" * 50)
    
    # 检查是否使用虚拟数据
    if args.use_dummy_data:
        print("使用虚拟数据进行测试...")
        
        # 创建虚拟模型（用于测试）
        class DummyModel(torch.nn.Module):
            def forward(self, batch):
                x, y = batch
                # 简单的线性变换作为虚拟预测
                pred = torch.randn_like(y)
                return pred, None
        
        model = DummyModel()
        sample_batch = create_dummy_data()
        
    else:
        # 尝试加载真实模型
        if not os.path.exists(args.model_path):
            print(f"错误: 模型文件不存在: {args.model_path}")
            print("请使用 --use_dummy_data 进行测试，或提供正确的模型路径")
            return
        
        try:
            model = load_model_simple(args.model_path)
            sample_batch = create_dummy_data()  # 暂时使用虚拟数据
            print("注意: 当前使用虚拟数据，请根据实际情况修改数据加载部分")
        except Exception as e:
            print(f"模型加载失败: {e}")
            print("请使用 --use_dummy_data 进行测试")
            return
    
    # 运行快速分析
    analyzer = QuickSensitivityAnalyzer(model, device='cpu')  # 使用CPU避免CUDA问题
    
    print("\n开始快速敏感性分析...")
    results = analyzer.quick_perturbation_test(sample_batch, args.save_dir)
    
    print("\n分析完成！")
    print(f"结果保存在: {args.save_dir}")
    
    # 输出排序结果
    print("\n变量重要性排序:")
    sorted_results = sorted(results.items(), key=lambda x: abs(x[1]), reverse=True)
    for i, (var_name, change) in enumerate(sorted_results, 1):
        importance = "高" if abs(change) > 5 else "中" if abs(change) > 1 else "低"
        print(f"{i}. {var_name}: {change:+.2f}% (重要性: {importance})")


if __name__ == '__main__':
    main()