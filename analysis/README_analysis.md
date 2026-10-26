# 海表变量影响机制分析工具

这个工具集用于分析不同海表变量对叶绿素浓度预测的影响机制，包含三种主要分析方法。

## 文件说明

- `main_analysis.py`: 主分析脚本，整合所有分析功能
- `attention_analysis.py`: 注意力机制分析
- `gradient_attribution.py`: 梯度归因分析  
- `sensitivity_analysis.py`: 敏感性分析
- `run_analysis.sh`: 便捷运行脚本
- `README_analysis.md`: 使用说明

## 分析方法

### 1. 敏感性分析 (Sensitivity Analysis)
- **扰动分析**: 对输入变量添加不同强度的噪声，观察预测性能变化
- **消融研究**: 逐个移除输入变量，评估每个变量的重要性

### 2. 梯度归因分析 (Gradient Attribution)
- **积分梯度**: 计算输入变量对预测结果的直接贡献
- **多深度分析**: 分析不同深度层的变量重要性模式

### 3. 注意力机制分析 (Attention Analysis)
- **注意力权重提取**: 分析模型内部的注意力分布
- **时空模式分析**: 研究注意力在时间和空间维度的变化

## 使用方法

### 方法1: 使用便捷脚本
```bash
# 1. 修改 run_analysis.sh 中的路径参数
vim run_analysis.sh

# 2. 运行分析
bash run_analysis.sh
```

### 方法2: 直接运行Python脚本
```bash
python main_analysis.py \
    --model_path experiments/train_34/checkpoints/best.ckpt \
    --config cfg_transfer_learning.yaml \
    --save_dir analysis_results \
    --device cuda
```

## 参数说明

- `--model_path`: 训练好的模型checkpoint路径
- `--config`: 配置文件路径 (YAML格式)
- `--save_dir`: 结果保存目录 (默认: comprehensive_analysis)
- `--device`: 计算设备 (cuda/cpu, 默认: cuda)

## 输出结果

分析完成后会生成以下文件和目录：

```
analysis_results/
├── comprehensive_report.md          # 综合分析报告
├── analysis_summary.png            # 可视化总结
├── sensitivity_analysis/            # 敏感性分析结果
│   ├── sensitivity_curves.png      # 敏感性曲线
│   ├── sensitivity_heatmap.png     # 敏感性热力图
│   ├── ablation_study.png          # 消融研究结果
│   ├── sensitivity_report.txt      # 敏感性分析报告
│   └── ablation_report.txt         # 消融研究报告
├── gradient_analysis/               # 梯度归因分析结果
│   ├── IntegratedGradients_analysis_depth_*.png
│   ├── depth_comparison_analysis.png
│   └── *.npy                       # 数值结果
└── attention_analysis/              # 注意力分析结果
    ├── attention_heatmap.png
    ├── importance_scores.png
    └── variable_importance_scores.npy
```

## 主要功能

### 1. 变量重要性排序
- 基于多种分析方法的综合评估
- 量化每个海表变量的贡献度
- 提供重要性等级分类

### 2. 深度依赖性分析
- 分析不同深度层对海表变量的敏感性
- 识别深度相关的影响模式
- 优化深度特异性预测策略

### 3. 时空模式识别
- 分析变量重要性的时间演化
- 识别空间分布特征
- 指导数据采集和预测策略

## 注意事项

1. **内存需求**: 分析过程需要较大内存，建议使用GPU加速
2. **数据格式**: 确保模型和数据格式兼容
3. **配置文件**: 使用与训练时相同的配置文件
4. **模型路径**: 确保模型checkpoint文件存在且可访问

## 故障排除

### 常见问题

1. **CUDA内存不足**
   ```bash
   # 减少batch_size或使用CPU
   python main_analysis.py --device cpu
   ```

2. **模型加载失败**
   ```bash
   # 检查模型路径和配置文件是否正确
   ls -la experiments/train_34/checkpoints/
   ```

3. **注意力分析失败**
   - 某些模型可能不包含显式注意力层
   - 这种情况下会跳过注意力分析，不影响其他分析

### 调试模式
```bash
# 添加详细日志输出
python main_analysis.py --model_path ... --config ... --save_dir ... 2>&1 | tee debug.log
```

## 结果解读

### 重要性得分
- **高重要性** (>5% MSE变化): 关键变量，对预测结果影响显著
- **中重要性** (1-5% MSE变化): 重要变量，有一定影响
- **低重要性** (<1% MSE变化): 次要变量，影响较小

### 分析建议
1. **数据收集**: 优先保证高重要性变量的数据质量
2. **模型优化**: 针对重要变量设计特殊的特征提取模块
3. **预测策略**: 在变量缺失时，重点关注高重要性变量的补全

## 扩展功能

可以根据需要添加更多分析方法：
- SHAP值分析
- LIME解释
- 特征交互分析
- 时间序列因果分析

## 联系方式

如有问题或建议，请联系开发团队。