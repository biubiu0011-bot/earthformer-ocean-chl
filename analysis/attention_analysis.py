"""
注意力机制分析：分析海表变量对不同深度叶绿素预测的影响
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os

class AttentionAnalyzer:
    def __init__(self, model, device='cuda'):
        self.model = model
        self.device = device
        self.model.eval()
        
        # 海表变量名称
        self.surface_vars = ['chla', 'sst', 'ssh', 'sss', 'ssu', 'ssv']
        self.depths = list(range(20))  # 20个深度层
        
    def extract_attention_weights(self, input_batch):
        """提取模型的注意力权重"""
        attention_weights = {}
        
        def hook_fn(name):
            def hook(module, input, output):
                if hasattr(output, 'attention_weights'):
                    attention_weights[name] = output.attention_weights.detach().cpu()
                elif isinstance(output, tuple) and len(output) > 1:
                    # 有些注意力层返回 (output, attention_weights)
                    attention_weights[name] = output[1].detach().cpu()
            return hook
        
        # 注册钩子函数到注意力层
        hooks = []
        for name, module in self.model.named_modules():
            if 'attention' in name.lower() or 'attn' in name.lower():
                hook = module.register_forward_hook(hook_fn(name))
                hooks.append(hook)
        
        # 前向传播
        with torch.no_grad():
            _ = self.model(input_batch)
        
        # 移除钩子
        for hook in hooks:
            hook.remove()
            
        return attention_weights
    
    def analyze_variable_importance(self, dataloader, save_dir='attention_analysis'):
        """分析不同海表变量的重要性"""
        os.makedirs(save_dir, exist_ok=True)
        
        all_attention_weights = []
        
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= 50:  # 只分析前50个batch
                break
                
            x, y = batch
            x = x.to(self.device)
            
            # 提取注意力权重
            attention_weights = self.extract_attention_weights((x, y))
            all_attention_weights.append(attention_weights)
            
            if batch_idx % 10 == 0:
                print(f"处理batch {batch_idx}/50")
        
        # 分析结果
        self._plot_attention_heatmap(all_attention_weights, save_dir)
        self._calculate_variable_importance(all_attention_weights, save_dir)
    
    def _plot_attention_heatmap(self, attention_weights_list, save_dir):
        """绘制注意力热力图"""
        # 这里需要根据你的模型具体结构来调整
        # 示例代码框架
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        for i, var_name in enumerate(self.surface_vars):
            ax = axes[i//3, i%3]
            
            # 计算该变量对所有深度的平均注意力权重
            # 这里需要根据实际的attention_weights结构来实现
            attention_matrix = np.random.rand(20, 14)  # 示例：20深度 x 14时间步
            
            sns.heatmap(attention_matrix, ax=ax, cmap='viridis',
                       xticklabels=range(14), yticklabels=range(20))
            ax.set_title(f'{var_name} Attention Weights')
            ax.set_xlabel('Time Steps')
            ax.set_ylabel('Depth Layers')
        
        plt.tight_layout()
        plt.savefig(f'{save_dir}/attention_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _calculate_variable_importance(self, attention_weights_list, save_dir):
        """计算变量重要性得分"""
        importance_scores = {}
        
        for var_idx, var_name in enumerate(self.surface_vars):
            # 计算该变量的平均重要性得分
            # 这里需要根据实际模型结构实现
            scores_by_depth = np.random.rand(20)  # 示例
            importance_scores[var_name] = scores_by_depth
        
        # 保存结果
        np.save(f'{save_dir}/variable_importance_scores.npy', importance_scores)
        
        # 绘制重要性得分图
        self._plot_importance_scores(importance_scores, save_dir)
    
    def _plot_importance_scores(self, importance_scores, save_dir):
        """绘制变量重要性得分"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 1. 按深度显示各变量重要性
        for var_name, scores in importance_scores.items():
            ax1.plot(range(20), scores, marker='o', label=var_name, linewidth=2)
        
        ax1.set_xlabel('Depth Layer')
        ax1.set_ylabel('Importance Score')
        ax1.set_title('Variable Importance by Depth')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. 总体重要性排序
        total_importance = {var: np.mean(scores) for var, scores in importance_scores.items()}
        vars_sorted = sorted(total_importance.items(), key=lambda x: x[1], reverse=True)
        
        vars_names = [item[0] for item in vars_sorted]
        vars_scores = [item[1] for item in vars_sorted]
        
        bars = ax2.bar(vars_names, vars_scores, color='skyblue', edgecolor='navy')
        ax2.set_xlabel('Surface Variables')
        ax2.set_ylabel('Average Importance Score')
        ax2.set_title('Overall Variable Importance Ranking')
        ax2.tick_params(axis='x', rotation=45)
        
        # 添加数值标签
        for bar, score in zip(bars, vars_scores):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{score:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(f'{save_dir}/importance_scores.png', dpi=300, bbox_inches='tight')
        plt.close()