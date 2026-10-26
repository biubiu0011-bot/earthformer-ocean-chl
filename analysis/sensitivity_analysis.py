"""
敏感性分析：通过扰动输入变量分析模型对不同海表变量的敏感性
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error
import os
from tqdm import tqdm

class SensitivityAnalyzer:
    def __init__(self, model, device='cuda'):
        self.model = model
        self.device = device
        self.model.eval()
        
        # 海表变量名称
        self.surface_vars = ['chla', 'sst', 'ssh', 'sss', 'ssu', 'ssv']
        
    def perturbation_analysis(self, dataloader, perturbation_levels=[0.1, 0.2, 0.5, 1.0], 
                            save_dir='sensitivity_analysis'):
        """
        扰动分析：逐个扰动每个输入变量，观察对预测结果的影响
        
        Args:
            dataloader: 测试数据加载器
            perturbation_levels: 扰动强度列表（相对于标准差的倍数）
            save_dir: 结果保存目录
        """
        os.makedirs(save_dir, exist_ok=True)
        
        print("开始敏感性分析...")
        
        # 存储结果
        sensitivity_results = {
            'baseline': {'mse': [], 'mae': []},
        }
        
        # 为每个变量和扰动级别初始化结果存储
        for var_name in self.surface_vars:
            sensitivity_results[var_name] = {}
            for level in perturbation_levels:
                sensitivity_results[var_name][level] = {'mse': [], 'mae': []}
        
        # 计算输入数据的统计信息（用于扰动）
        input_stats = self._calculate_input_statistics(dataloader)
        
        # 分析样本
        num_batches = min(20, len(dataloader))  # 限制分析的batch数量
        
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="敏感性分析")):
            if batch_idx >= num_batches:
                break
                
            x, y = batch
            x = x.to(self.device)
            y = y.to(self.device)
            
            # 1. 基线预测（无扰动）
            with torch.no_grad():
                baseline_pred, _ = self.model((x, y))
                baseline_mse = F.mse_loss(baseline_pred, y).item()
                baseline_mae = F.l1_loss(baseline_pred, y).item()
                
                sensitivity_results['baseline']['mse'].append(baseline_mse)
                sensitivity_results['baseline']['mae'].append(baseline_mae)
            
            # 2. 逐个变量扰动分析
            for var_idx, var_name in enumerate(self.surface_vars):
                if var_idx >= x.shape[-1]:  # 确保不超出通道数
                    continue
                    
                for level in perturbation_levels:
                    # 创建扰动后的输入
                    x_perturbed = x.clone()
                    
                    # 添加高斯噪声扰动
                    noise_std = level * input_stats['std'][var_idx]
                    noise = torch.randn_like(x_perturbed[..., var_idx]) * noise_std
                    x_perturbed[..., var_idx] += noise
                    
                    # 预测
                    with torch.no_grad():
                        perturbed_pred, _ = self.model((x_perturbed, y))
                        perturbed_mse = F.mse_loss(perturbed_pred, y).item()
                        perturbed_mae = F.l1_loss(perturbed_pred, y).item()
                        
                        sensitivity_results[var_name][level]['mse'].append(perturbed_mse)
                        sensitivity_results[var_name][level]['mae'].append(perturbed_mae)
        
        # 分析和可视化结果
        self._analyze_sensitivity_results(sensitivity_results, perturbation_levels, save_dir)
        
        return sensitivity_results
    
    def _calculate_input_statistics(self, dataloader):
        """计算输入数据的统计信息"""
        print("计算输入数据统计信息...")
        
        all_data = []
        for batch_idx, (x, _) in enumerate(dataloader):
            if batch_idx >= 10:  # 只用前10个batch计算统计信息
                break
            all_data.append(x.numpy())
        
        all_data = np.concatenate(all_data, axis=0)  # (N, T, H, W, C)
        
        # 计算每个通道的均值和标准差
        stats = {
            'mean': np.mean(all_data, axis=(0, 1, 2, 3)),  # (C,)
            'std': np.std(all_data, axis=(0, 1, 2, 3))     # (C,)
        }
        
        print("输入统计信息:")
        for i, var_name in enumerate(self.surface_vars):
            if i < len(stats['mean']):
                print(f"  {var_name}: mean={stats['mean'][i]:.4f}, std={stats['std'][i]:.4f}")
        
        return stats
    
    def _analyze_sensitivity_results(self, results, perturbation_levels, save_dir):
        """分析和可视化敏感性结果"""
        
        # 计算平均指标
        avg_results = {}
        for key, metrics in results.items():
            if key == 'baseline':
                avg_results[key] = {
                    'mse': np.mean(metrics['mse']),
                    'mae': np.mean(metrics['mae'])
                }
            else:
                avg_results[key] = {}
                for level in perturbation_levels:
                    avg_results[key][level] = {
                        'mse': np.mean(metrics[level]['mse']),
                        'mae': np.mean(metrics[level]['mae'])
                    }
        
        # 绘制结果
        self._plot_sensitivity_curves(avg_results, perturbation_levels, save_dir)
        self._plot_sensitivity_heatmap(avg_results, perturbation_levels, save_dir)
        
        # 保存数值结果
        np.save(f'{save_dir}/sensitivity_results.npy', avg_results)
        
        # 生成报告
        self._generate_sensitivity_report(avg_results, perturbation_levels, save_dir)
    
    def _plot_sensitivity_curves(self, avg_results, perturbation_levels, save_dir):
        """绘制敏感性曲线"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        baseline_mse = avg_results['baseline']['mse']
        baseline_mae = avg_results['baseline']['mae']
        
        # MSE曲线
        for var_name in self.surface_vars:
            if var_name in avg_results:
                mse_values = [avg_results[var_name][level]['mse'] for level in perturbation_levels]
                mse_relative = [(mse - baseline_mse) / baseline_mse * 100 for mse in mse_values]
                ax1.plot(perturbation_levels, mse_relative, 'o-', label=var_name, linewidth=2, markersize=6)
        
        ax1.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax1.set_xlabel('Perturbation Level (× std)')
        ax1.set_ylabel('MSE Change (%)')
        ax1.set_title('Model Sensitivity - MSE')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # MAE曲线
        for var_name in self.surface_vars:
            if var_name in avg_results:
                mae_values = [avg_results[var_name][level]['mae'] for level in perturbation_levels]
                mae_relative = [(mae - baseline_mae) / baseline_mae * 100 for mae in mae_values]
                ax2.plot(perturbation_levels, mae_relative, 'o-', label=var_name, linewidth=2, markersize=6)
        
        ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax2.set_xlabel('Perturbation Level (× std)')
        ax2.set_ylabel('MAE Change (%)')
        ax2.set_title('Model Sensitivity - MAE')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{save_dir}/sensitivity_curves.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_sensitivity_heatmap(self, avg_results, perturbation_levels, save_dir):
        """绘制敏感性热力图"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        baseline_mse = avg_results['baseline']['mse']
        baseline_mae = avg_results['baseline']['mae']
        
        # 准备数据
        mse_matrix = []
        mae_matrix = []
        
        for var_name in self.surface_vars:
            if var_name in avg_results:
                mse_row = []
                mae_row = []
                for level in perturbation_levels:
                    mse_change = (avg_results[var_name][level]['mse'] - baseline_mse) / baseline_mse * 100
                    mae_change = (avg_results[var_name][level]['mae'] - baseline_mae) / baseline_mae * 100
                    mse_row.append(mse_change)
                    mae_row.append(mae_change)
                mse_matrix.append(mse_row)
                mae_matrix.append(mae_row)
        
        mse_matrix = np.array(mse_matrix)
        mae_matrix = np.array(mae_matrix)
        
        # MSE热力图
        sns.heatmap(mse_matrix, 
                   xticklabels=[f'{level}×std' for level in perturbation_levels],
                   yticklabels=self.surface_vars,
                   annot=True, fmt='.2f', cmap='RdYlBu_r', center=0,
                   ax=ax1, cbar_kws={'label': 'MSE Change (%)'})
        ax1.set_title('MSE Sensitivity Heatmap')
        ax1.set_xlabel('Perturbation Level')
        ax1.set_ylabel('Surface Variables')
        
        # MAE热力图
        sns.heatmap(mae_matrix,
                   xticklabels=[f'{level}×std' for level in perturbation_levels],
                   yticklabels=self.surface_vars,
                   annot=True, fmt='.2f', cmap='RdYlBu_r', center=0,
                   ax=ax2, cbar_kws={'label': 'MAE Change (%)'})
        ax2.set_title('MAE Sensitivity Heatmap')
        ax2.set_xlabel('Perturbation Level')
        ax2.set_ylabel('Surface Variables')
        
        plt.tight_layout()
        plt.savefig(f'{save_dir}/sensitivity_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _generate_sensitivity_report(self, avg_results, perturbation_levels, save_dir):
        """生成敏感性分析报告"""
        report_path = f'{save_dir}/sensitivity_report.txt'
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("海表变量敏感性分析报告\n")
            f.write("=" * 50 + "\n\n")
            
            baseline_mse = avg_results['baseline']['mse']
            baseline_mae = avg_results['baseline']['mae']
            
            f.write(f"基线性能:\n")
            f.write(f"  MSE: {baseline_mse:.6f}\n")
            f.write(f"  MAE: {baseline_mae:.6f}\n\n")
            
            # 计算每个变量的平均敏感性
            f.write("变量敏感性排序 (基于最大扰动级别的MSE变化):\n")
            f.write("-" * 50 + "\n")
            
            sensitivity_scores = {}
            max_level = max(perturbation_levels)
            
            for var_name in self.surface_vars:
                if var_name in avg_results:
                    mse_change = (avg_results[var_name][max_level]['mse'] - baseline_mse) / baseline_mse * 100
                    sensitivity_scores[var_name] = mse_change
            
            # 按敏感性排序
            sorted_vars = sorted(sensitivity_scores.items(), key=lambda x: abs(x[1]), reverse=True)
            
            for i, (var_name, score) in enumerate(sorted_vars, 1):
                f.write(f"{i}. {var_name}: {score:+.2f}% MSE变化\n")
            
            f.write("\n详细结果:\n")
            f.write("-" * 50 + "\n")
            
            for var_name in self.surface_vars:
                if var_name in avg_results:
                    f.write(f"\n{var_name}:\n")
                    for level in perturbation_levels:
                        mse_change = (avg_results[var_name][level]['mse'] - baseline_mse) / baseline_mse * 100
                        mae_change = (avg_results[var_name][level]['mae'] - baseline_mae) / baseline_mae * 100
                        f.write(f"  扰动 {level}×std: MSE {mse_change:+.2f}%, MAE {mae_change:+.2f}%\n")
        
        print(f"敏感性分析报告已保存到: {report_path}")
    
    def ablation_study(self, dataloader, save_dir='sensitivity_analysis'):
        """
        消融研究：逐个移除输入变量，观察对模型性能的影响
        """
        os.makedirs(save_dir, exist_ok=True)
        
        print("开始消融研究...")
        
        ablation_results = {
            'baseline': {'mse': [], 'mae': []},
        }
        
        # 为每个变量初始化结果存储
        for var_name in self.surface_vars:
            ablation_results[f'without_{var_name}'] = {'mse': [], 'mae': []}
        
        num_batches = min(20, len(dataloader))
        
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="消融研究")):
            if batch_idx >= num_batches:
                break
                
            x, y = batch
            x = x.to(self.device)
            y = y.to(self.device)
            
            # 基线预测
            with torch.no_grad():
                baseline_pred, _ = self.model((x, y))
                baseline_mse = F.mse_loss(baseline_pred, y).item()
                baseline_mae = F.l1_loss(baseline_pred, y).item()
                
                ablation_results['baseline']['mse'].append(baseline_mse)
                ablation_results['baseline']['mae'].append(baseline_mae)
            
            # 逐个移除变量
            for var_idx, var_name in enumerate(self.surface_vars):
                if var_idx >= x.shape[-1]:
                    continue
                    
                # 创建移除该变量的输入（设为0）
                x_ablated = x.clone()
                x_ablated[..., var_idx] = 0
                
                with torch.no_grad():
                    ablated_pred, _ = self.model((x_ablated, y))
                    ablated_mse = F.mse_loss(ablated_pred, y).item()
                    ablated_mae = F.l1_loss(ablated_pred, y).item()
                    
                    ablation_results[f'without_{var_name}']['mse'].append(ablated_mse)
                    ablation_results[f'without_{var_name}']['mae'].append(ablated_mae)
        
        # 分析消融结果
        self._analyze_ablation_results(ablation_results, save_dir)
        
        return ablation_results
    
    def _analyze_ablation_results(self, results, save_dir):
        """分析消融研究结果"""
        
        # 计算平均指标
        avg_results = {}
        for key, metrics in results.items():
            avg_results[key] = {
                'mse': np.mean(metrics['mse']),
                'mae': np.mean(metrics['mae'])
            }
        
        # 绘制结果
        self._plot_ablation_results(avg_results, save_dir)
        
        # 保存结果
        np.save(f'{save_dir}/ablation_results.npy', avg_results)
        
        # 生成报告
        self._generate_ablation_report(avg_results, save_dir)
    
    def _plot_ablation_results(self, avg_results, save_dir):
        """绘制消融研究结果"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        baseline_mse = avg_results['baseline']['mse']
        baseline_mae = avg_results['baseline']['mae']
        
        var_names = []
        mse_changes = []
        mae_changes = []
        
        for var_name in self.surface_vars:
            key = f'without_{var_name}'
            if key in avg_results:
                var_names.append(var_name)
                mse_change = (avg_results[key]['mse'] - baseline_mse) / baseline_mse * 100
                mae_change = (avg_results[key]['mae'] - baseline_mae) / baseline_mae * 100
                mse_changes.append(mse_change)
                mae_changes.append(mae_change)
        
        # MSE变化
        bars1 = ax1.bar(var_names, mse_changes, color='lightcoral', edgecolor='darkred')
        ax1.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax1.set_ylabel('MSE Change (%)')
        ax1.set_title('Ablation Study - MSE Impact')
        ax1.tick_params(axis='x', rotation=45)
        
        # 添加数值标签
        for bar, change in zip(bars1, mse_changes):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + (max(mse_changes) - min(mse_changes))*0.01,
                    f'{change:+.2f}%', ha='center', va='bottom' if height >= 0 else 'top')
        
        # MAE变化
        bars2 = ax2.bar(var_names, mae_changes, color='lightblue', edgecolor='darkblue')
        ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax2.set_ylabel('MAE Change (%)')
        ax2.set_title('Ablation Study - MAE Impact')
        ax2.tick_params(axis='x', rotation=45)
        
        # 添加数值标签
        for bar, change in zip(bars2, mae_changes):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + (max(mae_changes) - min(mae_changes))*0.01,
                    f'{change:+.2f}%', ha='center', va='bottom' if height >= 0 else 'top')
        
        plt.tight_layout()
        plt.savefig(f'{save_dir}/ablation_study.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _generate_ablation_report(self, avg_results, save_dir):
        """生成消融研究报告"""
        report_path = f'{save_dir}/ablation_report.txt'
        
        baseline_mse = avg_results['baseline']['mse']
        baseline_mae = avg_results['baseline']['mae']
        
        # 计算重要性得分
        importance_scores = {}
        for var_name in self.surface_vars:
            key = f'without_{var_name}'
            if key in avg_results:
                # 移除变量后性能下降越多，说明该变量越重要
                mse_change = (avg_results[key]['mse'] - baseline_mse) / baseline_mse * 100
                importance_scores[var_name] = mse_change
        
        # 按重要性排序（性能下降最多的最重要）
        sorted_vars = sorted(importance_scores.items(), key=lambda x: x[1], reverse=True)
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("消融研究报告\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"基线性能:\n")
            f.write(f"  MSE: {baseline_mse:.6f}\n")
            f.write(f"  MAE: {baseline_mae:.6f}\n\n")
            
            f.write("变量重要性排序 (基于移除后的性能变化):\n")
            f.write("-" * 50 + "\n")
            
            for i, (var_name, score) in enumerate(sorted_vars, 1):
                key = f'without_{var_name}'
                mse_change = score
                mae_change = (avg_results[key]['mae'] - baseline_mae) / baseline_mae * 100
                
                f.write(f"{i}. {var_name}:\n")
                f.write(f"   移除后MSE变化: {mse_change:+.2f}%\n")
                f.write(f"   移除后MAE变化: {mae_change:+.2f}%\n")
                f.write(f"   重要性: {'高' if mse_change > 5 else '中' if mse_change > 1 else '低'}\n\n")
        
        print(f"消融研究报告已保存到: {report_path}")


# 导入必要的模块
import torch.nn.functional as F