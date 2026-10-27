# Earthformer-Ocean-Chl

> English version: [README_EN.md](README_EN.md)

基于深度迁移学习的西北太平洋逐日尺度三维叶绿素 a（Chl-a）场重构项目。

本项目在 [Earthformer](https://github.com/amazon-science/earth-forecasting-transformer) 上进行二次开发：以连续 14 天的海表变量为输入，先在 NEMO 三维 Chl-a 场上预训练，再迁移至稀疏 BGC-Argo 浮标剖面，重构 0–300 m 的 20 层逐日三维 Chl-a 场。

> 这是可公开的代码整理版。原始卫星/NEMO/BGC-Argo 数据、模型权重、训练日志和实验结果均不随仓库发布；首次复现前请阅读 [数据说明](data/README.md)、[迁移学习说明](docs/transfer_learning.md)，并核对数据变量顺序、深度层、海陆掩码和归一化统计。

## 研究流程

```text
数据处理 → NEMO 预训练 → Argo 迁移 → 对比实验 → 可解释性
```

1. **数据处理**：融合表层 Chl-a、SST、SSS、SSH、风场、NEMO 和 BGC-Argo 数据，执行质量控制、掩码、时空匹配与标准化。
2. **NEMO 预训练**：以完整的 NEMO 三维场为监督信号，训练 Earthformer-ISTRB 学习时空与垂向结构。
3. **Argo 迁移**：加载 NEMO 预训练权重，冻结编码器并微调解码器，适配 BGC-Argo 实测剖面。
4. **对比实验**：与 CNN、ConvLSTM、ViT 对比；包含损失函数与残差模块消融。
5. **可解释性**：使用梯度归因、敏感性分析和注意力分析，考察不同海表变量的贡献。

## 方法概览

- 输入：`[batch, 14 days, 45, 60, 6 variables]`
- 输入变量：表层 Chl-a、SST、SSS、SSH、风场 U、风场 V
- 输出：`[batch, 14 days, 45, 60, 20 depth layers]`
- 目标：0–300 m 三维 Chl-a
- 主模型：Earthformer + 改进时空残差块（ISTRB，2D+1D 分解）
- 损失：Hybrid Scale-aware Loss（Huber + MSE，`alpha=0.7`）

## 论文结果

| 模型（NEMO 测试集） | Pearson's r | NSE | RMSE (mg/m³) | MAE (mg/m³) |
|---|---:|---:|---:|---:|
| CNN | 0.8589 | 0.7364 | 0.0900 | 0.0483 |
| ConvLSTM | 0.8746 | 0.7641 | 0.0851 | 0.0446 |
| ViT | 0.8633 | 0.7440 | 0.0887 | 0.0451 |
| Earthformer-ISTRB | **0.8993** | **0.8056** | **0.0741** | **0.0368** |

Argo 迁移测试集：r = 0.8935、NSE = 0.7975、RMSE = 0.0722 mg/m³、MAE = 0.0343 mg/m³。与 125,380 个时空匹配 BGC-Argo 点外部检验：r = 0.8620、NSE = 0.6759。

## 仓库结构

```text
configs/     NEMO 预训练与 Argo 迁移配置
scripts/     数据处理、NEMO 训练、Argo 迁移入口
src/         Earthformer 上游实现与本项目数据模块/路径工具
baselines/   CNN、ConvLSTM、ViT 对比模型
analysis/    可解释性与结果分析脚本
notebooks/   本地保留的 Argo 探索记录；暂不提交，见 `notebooks/README.md`
docs/        方法、迁移学习与 AI 项目说明
data/        数据下载、许可和目录约定（不含数据）
results/     结果与权重发布约定（不含结果）
tests/       仓库结构与可移植性检查
```

## 关键代码

| 环节 | 当前仓库入口 |
|---|---|
| NEMO 预训练 | `scripts/train_nemo_pretrain.py` + `configs/nemo_pretrain.yaml` |
| Argo 迁移 | `scripts/train_argo_transfer.py` + `configs/argo_transfer.yaml` |
| NEMO / Argo 数据加载 | `src/earthformer_ocean_chl/data/nemo_dataset.py`、`argo_dataset.py` |
| Earthformer-ISTRB | `src/earthformer/cuboid_transformer/cuboid_transformer_update.py` |
| 改进残差块 | `src/earthformer/cuboid_transformer/residual_blocks.py` |
| Hybrid 损失 | `src/earthformer/losses/hybrid_scale_aware_loss.py` |
| 对比模型 | `baselines/` |
| 可解释性 | `analysis/` |

### Argo 数据处理脚本

| 目的 | 脚本 |
|---|---|
| 合并与标准化原始剖面 CSV | `scripts/merge_bgc_argo_profiles.py` |
| 0–300 m 插值与可选平滑 | `scripts/process_bgc_argo_profiles.py` |
| BGC-Argo 与 NEMO 最近邻匹配 | `scripts/match_argo_to_nemo.py` |
| 提取浮标元数据 | `scripts/extract_argo_metadata.py` |
| 写入已准备好的迁移学习张量 | `scripts/build_argo_transfer_samples.py` |
| 指标、剖面图、密度散点图 | `analysis/compare_argo_nemo.py`、`plot_argo_profiles.py`、`plot_density_scatter.py` |

所有脚本均支持 `--help`。迁移样本写入脚本只验证并保存已经完成网格化/归一化的张量；它不会擅自推断论文中的通道顺序、掩码或归一化统计。

## 安装

建议使用 Linux 或 WSL + NVIDIA GPU 环境。CUDA、PyTorch、PyTorch Lightning 与 Apex 需要彼此兼容。

```bash
conda env create -f environment.yml
conda activate earthformer-ocean-chl
pip install -e .
```

如果使用 Apex，请按本机 CUDA 与 PyTorch 版本单独编译安装；它不由本仓库自动安装。

## 数据与路径配置

数据不会上传到 GitHub。设置以下环境变量后再运行脚本：

```bash
# Linux / macOS
export EARTHFORMER_OCEAN_CHL_DATA_ROOT=/path/to/ocean-chl-data
export EARTHFORMER_OCEAN_CHL_RESULTS_ROOT=/path/to/ocean-chl-results

# PowerShell
$env:EARTHFORMER_OCEAN_CHL_DATA_ROOT = 'D:\path\to\ocean-chl-data'
$env:EARTHFORMER_OCEAN_CHL_RESULTS_ROOT = 'D:\path\to\ocean-chl-results'
```

NEMO 预训练加载器预期的数据目录与文件命名见 [data/README.md](data/README.md)。在运行前还须核对变量顺序、20 个深度层、海陆掩码和归一化统计是否与论文一致。

## 运行

先做结构检查：

```bash
python tests/test_repository_layout.py
```

NEMO 预训练：

```bash
python scripts/train_nemo_pretrain.py \
  --gpus 1 \
  --cfg configs/nemo_pretrain.yaml \
  --save nemo_pretrain
```

Argo 迁移（需替换为实际 NEMO checkpoint）：

```bash
python scripts/train_argo_transfer.py \
  --gpus 1 \
  --cfg configs/argo_transfer.yaml \
  --pretrain_ckpt /path/to/nemo_pretrain.ckpt \
  --save argo_transfer
```

训练输出默认写入 `EARTHFORMER_OCEAN_CHL_RESULTS_ROOT`（未设置时为仓库内 `results/`）。可解释性分析可通过 `analysis/run_analysis.sh` 启动；请先设置 `MODEL_PATH`，其余参数可用环境变量覆盖。详细迁移参数见 [docs/transfer_learning.md](docs/transfer_learning.md)。

> 这些命令给出当前仓库的真实入口，但不宣称可在未准备数据的情况下直接复现论文；请先用少量样本验证形状与单批推理。

## 可解释性结论

- SSH 跨深度影响最强，约 70–90 m 次表层最显著；
- SST 主要影响 10–50 m 浅层；
- 冬季模型更依赖物理动力变量，夏季相对更关注 Chl-a 自身；
- 高重要性区域与西北太平洋中高纬涡旋、锋面等高能动力区相符。

## 开源、数据与安全

- `src/earthformer/` 源自上游 Earthformer，保留 [Apache-2.0 许可证](LICENSE)及其版权声明。
- 发布数据、图件和权重前须确认数据许可、作者署名和再分发权限。
- 不要提交账号、密码、token、私有服务器地址、原始数据、checkpoint、训练日志或本机绝对路径。

## 引用

```text
李晓玲. 基于深度迁移学习的西北太平洋逐日尺度三维叶绿素场重构.
硕士学位论文, 青岛科技大学, 2026.

Gao, Z. et al. Earthformer: Exploring Space-Time Transformers for Earth System Forecasting.
NeurIPS, 2022.
```
