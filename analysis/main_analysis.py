"""
海表变量对叶绿素预测影响的综合分析主脚本
整合注意力分析、梯度归因分析和敏感性分析

使用方法:
python main_analysis.py --model_path experiments/train_34/checkpoints/best.ckpt --config cfg_transfer_learning.yaml
"""

import sys
import os

# === 修改开始：使用相对路径配置 ===
# 获取当前脚本的目录
current_dir = os.path.dirname(os.path.abspath(__file__))

# 查找 earth-forecasting-transformer-main 目录
earthformer_main_dir = None
for root, dirs, files in os.walk(current_dir):
    if 'earth-forecasting-transformer-main' in dirs:
        earthformer_main_dir = os.path.join(root, 'earth-forecasting-transformer-main')
        break

if earthformer_main_dir is None:
    # 如果在当前目录找不到，尝试在当前目录下直接查找
    if os.path.exists(os.path.join(current_dir, 'earth-forecasting-transformer-main')):
        earthformer_main_dir = os.path.join(current_dir, 'earth-forecasting-transformer-main')
    else:
        print("警告: 找不到 earth-forecasting-transformer-main 目录")
        print("请确保脚本在正确的位置运行")
        earthformer_main_dir = os.environ.get("EARTHFORMER_OCEAN_CHL_SOURCE_ROOT", "src")

print(f"使用 Earthformer 目录: {earthformer_main_dir}")

# 1. 添加 src 目录到路径
src_path = os.path.join(earthformer_main_dir, "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

# 2. 添加 scripts/cuboid_transformer/moving_mnist 目录到路径
script_path = os.path.join(earthformer_main_dir, "scripts", "cuboid_transformer", "moving_mnist")
if script_path not in sys.path:
    sys.path.insert(0, script_path)

# 3. 添加当前目录
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from omegaconf import OmegaConf
import warnings
warnings.filterwarnings('ignore')

# 导入分析模块
from attention_analysis import AttentionAnalyzer
from gradient_attribution import GradientAttributionAnalyzer
from sensitivity_analysis import SensitivityAnalyzer

# 导入模型和数据模块
try:
    from earthformer.datasets.moving_mnist.moving_mnist_transfer import MovingMNISTDataModule as TransferDataModule
except ImportError:
    print("警告: 无法导入 TransferDataModule，请检查路径")
    TransferDataModule = None

try:
    from train_cuboid_mnist_29 import CuboidMovingMNISTPLModule
except ImportError:
    print("警告: 无法导入 CuboidMovingMNISTPLModule，请检查路径")
    CuboidMovingMNISTPLModule = None


class ComprehensiveAnalyzer:
    """综合分析器：整合多种分析方法"""
    
    def __init__(self, model_path, config_path, device='cuda'):
        self.device = device
        self.model_path = model_path
        self.config_path = config_path
        
        # 加载配置和模型
        self.cfg = self._load_config()
        self.model = self._load_model()
        self.dataloader = self._prepare_dataloader()
        
        # 初始化分析器
        self.attention_analyzer = AttentionAnalyzer(self.model, device)
        self.gradient_analyzer = GradientAttributionAnalyzer(self.model, device)
        self.sensitivity_analyzer = SensitivityAnalyzer(self.model, device)
        
        # 海表变量名称
        self.surface_vars = ['chla', 'sst', 'ssh', 'sss', 'ssu', 'ssv']
        
    def _load_config(self):
        """加载配置文件"""
        print(f"加载配置文件: {self.config_path}")
        cfg = OmegaConf.load(self.config_path)
        return cfg
    
    def _load_model(self):
        """加载训练好的模型"""
        print(f"加载模型: {self.model_path}")
        
        if CuboidMovingMNISTPLModule is None:
            raise ImportError("无法导入模型类，请检查导入路径")
        
        try:
            # 方法1: 手动加载checkpoint，避免save_dir问题
            print("手动加载checkpoint...")
            checkpoint = torch.load(self.model_path, map_location='cpu')
            
            # 获取超参数
            if 'hyper_parameters' in checkpoint:
                hyper_params = checkpoint['hyper_parameters']
                print(f"找到超参数: {list(hyper_params.keys())}")
            else:
                hyper_params = {}
            
            # 创建模型实例，提供必要参数
            model = CuboidMovingMNISTPLModule(
                total_num_steps=hyper_params.get('total_num_steps', 1),
                save_dir="temp_analysis_dir",  # 提供临时目录避免None错误
                oc_file=None  # 不使用配置文件
            )
            
            # 加载状态字典
            model.load_state_dict(checkpoint['state_dict'], strict=False)
            print("✓ 模型权重加载成功")
            
        except Exception as e:
            print(f"手动加载失败: {e}")
            try:
                # 方法2: 使用load_from_checkpoint，但跳过save_dir验证
                print("尝试使用load_from_checkpoint...")
                
                # 临时创建目录
                temp_dir = "temp_analysis_dir"
                os.makedirs(temp_dir, exist_ok=True)
                
                model = CuboidMovingMNISTPLModule.load_from_checkpoint(
                    self.model_path,
                    total_num_steps=1,
                    save_dir=temp_dir,
                    strict=False
                )
                print("✓ load_from_checkpoint成功")
                
            except Exception as e2:
                print(f"所有方法都失败了: {e}, {e2}")
                print("请检查:")
                print("1. 模型文件是否存在且完整")
                print("2. 是否在正确的环境中运行")
                print("3. 依赖包是否正确安装")
                raise RuntimeError(f"无法加载模型: 尝试了多种方法都失败")
        
        model = model.to(self.device)
        model.eval()
        
        print("✓ 模型加载完成并设置为评估模式")
        return model
    
    def _prepare_dataloader(self):
        """准备测试数据加载器"""
        print("准备数据加载器...")
        
        if TransferDataModule is None:
            raise ImportError("无法导入数据模块，请检查导入路径")
        
        # 创建数据模块
        dm = TransferDataModule(
            dataset_name=self.cfg.dataset.dataset_name,
            dataset_dir=self.cfg.dataset.dataset_dir,
            batch_size=self.cfg.optim.batch_size,
            num_workers=self.cfg.dataset.num_workers,
            # 其他参数根据配置文件设置
        )
        
        dm.setup(stage='test')
        test_dataloader = dm.test_dataloader()
        
        print(f"数据加载器准备完成，测试集大小: {len(test_dataloader)}")
        return test_dataloader
    
    def run_comprehensive_analysis(self, save_dir='comprehensive_analysis'):
        """运行综合分析"""
        print("=" * 60)
        print("开始海表变量影响机制综合分析")
        print("=" * 60)
        
        # 创建保存目录
        os.makedirs(save_dir, exist_ok=True)
        
        results = {}
        
        # 1. 敏感性分析
        print("\n1. 敏感性分析...")
        print("-" * 40)
        
        sensitivity_dir = os.path.join(save_dir, 'sensitivity_analysis')
        
        # 扰动分析
        sensitivity_results = self.sensitivity_analyzer.perturbation_analysis(
            self.dataloader,
            perturbation_levels=[0.1, 0.2, 0.5, 1.0],
            save_dir=sensitivity_dir
        )
        
        # 消融研究
        ablation_results = self.sensitivity_analyzer.ablation_study(
            self.dataloader,
            save_dir=sensitivity_dir
        )
        
        results['sensitivity'] = {
            'perturbation': sensitivity_results,
            'ablation': ablation_results
        }
        
        # 2. 梯度归因分析
        print("\n2. 梯度归因分析...")
        print("-" * 40)
        
        gradient_dir = os.path.join(save_dir, 'gradient_analysis')
        
        # 获取样本batch进行分析
        sample_batch = next(iter(self.dataloader))
        
        # 分析多个深度层
        gradient_results = self.gradient_analyzer.analyze_multiple_depths(
            self.dataloader,
            depths_to_analyze=[0, 5, 10, 15, 19],
            save_dir=gradient_dir
        )
        
        results['gradient'] = gradient_results
        
        # 3. 注意力分析
        print("\n3. 注意力机制分析...")
        print("-" * 40)
        
        attention_dir = os.path.join(save_dir, 'attention_analysis')
        
        try:
            self.attention_analyzer.analyze_variable_importance(
                self.dataloader,
                save_dir=attention_dir
            )
            results['attention'] = 'completed'
        except Exception as e:
            print(f"注意力分析遇到问题: {e}")
            print("可能是模型结构不包含显式的注意力层")
            results['attention'] = 'failed'
        
        # 4. 生成综合报告
        print("\n4. 生成综合分析报告...")
        print("-" * 40)
        
        self._generate_comprehensive_report(results, save_dir)
        
        # 5. 创建可视化总结
        self._create_summary_visualization(results, save_dir)
        
        print("\n" + "=" * 60)
        print("综合分析完成！")
        print(f"结果保存在: {save_dir}")
        print("=" * 60)
        
        return results
    
    def _generate_comprehensive_report(self, results, save_dir):
        """生成综合分析报告"""
        report_path = os.path.join(save_dir, 'comprehensive_report.md')
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# 海表变量对叶绿素预测影响的综合分析报告\n\n")
            
            f.write("## 分析概述\n\n")
            f.write("本报告通过多种分析方法，系统评估了不同海表变量对叶绿素浓度预测的影响：\n\n")
            f.write("1. **敏感性分析**: 通过扰动和消融实验评估变量重要性\n")
            f.write("2. **梯度归因分析**: 使用积分梯度分析输入变量的贡献\n")
            f.write("3. **注意力机制分析**: 分析模型内部的注意力权重分布\n\n")
            
            # 敏感性分析结果
            if 'sensitivity' in results:
                f.write("## 敏感性分析结果\n\n")
                
                # 从消融研究中提取变量重要性排序
                ablation_results = results['sensitivity']['ablation']
                if 'baseline' in ablation_results:
                    baseline_mse = np.mean(ablation_results['baseline']['mse'])
                    
                    f.write("### 变量重要性排序（基于消融研究）\n\n")
                    
                    importance_scores = {}
                    for var_name in self.surface_vars:
                        key = f'without_{var_name}'
                        if key in ablation_results:
                            var_mse = np.mean(ablation_results[key]['mse'])
                            mse_change = (var_mse - baseline_mse) / baseline_mse * 100
                            importance_scores[var_name] = mse_change
                    
                    # 按重要性排序
                    sorted_vars = sorted(importance_scores.items(), key=lambda x: x[1], reverse=True)
                    
                    for i, (var_name, score) in enumerate(sorted_vars, 1):
                        importance_level = "高" if score > 5 else "中" if score > 1 else "低"
                        f.write(f"{i}. **{var_name}**: {score:+.2f}% MSE变化 (重要性: {importance_level})\n")
                    
                    f.write("\n")
            
            # 梯度归因分析结果
            if 'gradient' in results:
                f.write("## 梯度归因分析结果\n\n")
                f.write("通过积分梯度方法分析了不同深度层的变量重要性模式。\n\n")
                f.write("详细结果请查看 `gradient_analysis/` 目录中的可视化图表。\n\n")
            
            # 注意力分析结果
            if 'attention' in results:
                f.write("## 注意力机制分析结果\n\n")
                if results['attention'] == 'completed':
                    f.write("成功提取并分析了模型的注意力权重分布。\n\n")
                    f.write("详细结果请查看 `attention_analysis/` 目录中的可视化图表。\n\n")
                else:
                    f.write("注意力分析未能完成，可能是模型结构不包含显式的注意力层。\n\n")
            
            f.write("## 主要发现\n\n")
            f.write("1. **最重要的海表变量**: 根据多种分析方法的一致性结果确定\n")
            f.write("2. **深度依赖性**: 不同深度层对海表变量的敏感性存在差异\n")
            f.write("3. **时空模式**: 变量重要性在时间和空间维度上的分布特征\n\n")
            
            f.write("## 建议\n\n")
            f.write("1. **数据收集优先级**: 重点关注高重要性的海表变量\n")
            f.write("2. **模型改进方向**: 针对重要变量优化特征提取\n")
            f.write("3. **预测策略**: 根据变量可用性调整预测方法\n\n")
            
            f.write("---\n")
            f.write("*报告生成时间: 自动生成*\n")
        
        print(f"综合报告已保存到: {report_path}")
    
    def _create_summary_visualization(self, results, save_dir):
        """创建分析结果的可视化总结"""
        
        if 'sensitivity' not in results:
            print("缺少敏感性分析结果，跳过可视化总结")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. 消融研究结果
        ablation_results = results['sensitivity']['ablation']
        if 'baseline' in ablation_results:
            baseline_mse = np.mean(ablation_results['baseline']['mse'])
            
            var_names = []
            mse_changes = []
            
            for var_name in self.surface_vars:
                key = f'without_{var_name}'
                if key in ablation_results:
                    var_names.append(var_name)
                    var_mse = np.mean(ablation_results[key]['mse'])
                    mse_change = (var_mse - baseline_mse) / baseline_mse * 100
                    mse_changes.append(mse_change)
            
            bars = axes[0, 0].bar(var_names, mse_changes, color='lightcoral', edgecolor='darkred')
            axes[0, 0].axhline(y=0, color='black', linestyle='--', alpha=0.5)
            axes[0, 0].set_title('变量重要性 (消融研究)', fontsize=14, fontweight='bold')
            axes[0, 0].set_ylabel('MSE变化 (%)')
            axes[0, 0].tick_params(axis='x', rotation=45)
            
            # 添加数值标签
            for bar, change in zip(bars, mse_changes):
                height = bar.get_height()
                axes[0, 0].text(bar.get_x() + bar.get_width()/2., height + max(mse_changes)*0.01,
                               f'{change:+.1f}%', ha='center', va='bottom' if height >= 0 else 'top')
        
        # 2. 扰动分析结果（选择最大扰动级别）
        perturbation_results = results['sensitivity']['perturbation']
        if 'baseline' in perturbation_results:
            baseline_mse = np.mean(perturbation_results['baseline']['mse'])
            max_level = 1.0  # 使用最大扰动级别
            
            var_names = []
            sensitivity_scores = []
            
            for var_name in self.surface_vars:
                if var_name in perturbation_results and max_level in perturbation_results[var_name]:
                    var_names.append(var_name)
                    perturbed_mse = np.mean(perturbation_results[var_name][max_level]['mse'])
                    sensitivity = (perturbed_mse - baseline_mse) / baseline_mse * 100
                    sensitivity_scores.append(sensitivity)
            
            bars = axes[0, 1].bar(var_names, sensitivity_scores, color='lightblue', edgecolor='darkblue')
            axes[0, 1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
            axes[0, 1].set_title('变量敏感性 (扰动分析)', fontsize=14, fontweight='bold')
            axes[0, 1].set_ylabel('MSE变化 (%)')
            axes[0, 1].tick_params(axis='x', rotation=45)
            
            for bar, score in zip(bars, sensitivity_scores):
                height = bar.get_height()
                axes[0, 1].text(bar.get_x() + bar.get_width()/2., height + max(sensitivity_scores)*0.01,
                               f'{score:+.1f}%', ha='center', va='bottom' if height >= 0 else 'top')
        
        # 3. 方法对比
        if len(var_names) > 0:
            x = np.arange(len(var_names))
            width = 0.35
            
            bars1 = axes[1, 0].bar(x - width/2, mse_changes, width, label='消融研究', 
                                  color='lightcoral', edgecolor='darkred')
            bars2 = axes[1, 0].bar(x + width/2, sensitivity_scores, width, label='扰动分析',
                                  color='lightblue', edgecolor='darkblue')
            
            axes[1, 0].axhline(y=0, color='black', linestyle='--', alpha=0.5)
            axes[1, 0].set_title('分析方法对比', fontsize=14, fontweight='bold')
            axes[1, 0].set_ylabel('MSE变化 (%)')
            axes[1, 0].set_xticks(x)
            axes[1, 0].set_xticklabels(var_names, rotation=45)
            axes[1, 0].legend()
        
        # 4. 重要性排序总结
        if len(var_names) > 0:
            # 计算综合重要性得分（消融研究权重更高）
            combined_scores = {}
            for i, var_name in enumerate(var_names):
                # 消融研究权重0.7，扰动分析权重0.3
                combined_score = 0.7 * abs(mse_changes[i]) + 0.3 * abs(sensitivity_scores[i])
                combined_scores[var_name] = combined_score
            
            # 排序
            sorted_vars = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
            
            var_names_sorted = [item[0] for item in sorted_vars]
            scores_sorted = [item[1] for item in sorted_vars]
            
            bars = axes[1, 1].barh(var_names_sorted, scores_sorted, color='gold', edgecolor='orange')
            axes[1, 1].set_title('综合重要性排序', fontsize=14, fontweight='bold')
            axes[1, 1].set_xlabel('综合重要性得分')
            
            # 添加排名标签
            for i, (bar, score) in enumerate(zip(bars, scores_sorted)):
                axes[1, 1].text(score + max(scores_sorted)*0.01, bar.get_y() + bar.get_height()/2,
                               f'#{i+1}', ha='left', va='center', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'analysis_summary.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"可视化总结已保存到: {os.path.join(save_dir, 'analysis_summary.png')}")


def main():
    parser = argparse.ArgumentParser(description='海表变量影响机制综合分析')
    parser.add_argument('--model_path', type=str, required=True,
                       help='训练好的模型checkpoint路径')
    parser.add_argument('--config', type=str, required=True,
                       help='配置文件路径')
    parser.add_argument('--save_dir', type=str, default='comprehensive_analysis',
                       help='结果保存目录')
    parser.add_argument('--device', type=str, default='cuda',
                       help='计算设备')
    
    args = parser.parse_args()
    
    # 检查文件是否存在
    if not os.path.exists(args.model_path):
        print(f"错误: 模型文件不存在: {args.model_path}")
        return
    
    if not os.path.exists(args.config):
        print(f"错误: 配置文件不存在: {args.config}")
        return
    
    # 创建分析器并运行分析
    analyzer = ComprehensiveAnalyzer(
        model_path=args.model_path,
        config_path=args.config,
        device=args.device
    )
    
    # 运行综合分析
    results = analyzer.run_comprehensive_analysis(save_dir=args.save_dir)
    
    print("\n分析完成！主要结果文件：")
    print(f"- 综合报告: {args.save_dir}/comprehensive_report.md")
    print(f"- 可视化总结: {args.save_dir}/analysis_summary.png")
    print(f"- 敏感性分析: {args.save_dir}/sensitivity_analysis/")
    print(f"- 梯度归因分析: {args.save_dir}/gradient_analysis/")
    print(f"- 注意力分析: {args.save_dir}/attention_analysis/")


if __name__ == '__main__':
    main()
