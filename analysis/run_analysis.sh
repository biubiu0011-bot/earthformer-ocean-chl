#!/bin/bash

# Run from any directory with: bash analysis/run_analysis.sh
# Required: export MODEL_PATH=/path/to/checkpoint.ckpt

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "开始海表变量影响机制综合分析..."

# Environment variables override these portable defaults.
: "${MODEL_PATH:?Set MODEL_PATH to an Argo-transfer checkpoint before running this script.}"
CONFIG_PATH="${CONFIG_PATH:-$REPO_ROOT/configs/argo_transfer.yaml}"
RESULTS_ROOT="${EARTHFORMER_OCEAN_CHL_RESULTS_ROOT:-$REPO_ROOT/results}"
SAVE_DIR="${SAVE_DIR:-$RESULTS_ROOT/analysis_$(date +%Y%m%d_%H%M%S)}"
DEVICE="${DEVICE:-cuda}"

# 检查文件是否存在
if [ ! -f "$MODEL_PATH" ]; then
    echo "错误: 模型文件不存在: $MODEL_PATH"
    echo "请修改 MODEL_PATH 为正确的模型路径"
    exit 1
fi

if [ ! -f "$CONFIG_PATH" ]; then
    echo "错误: 配置文件不存在: $CONFIG_PATH"
    echo "请修改 CONFIG_PATH 为正确的配置文件路径"
    exit 1
fi

# 创建结果目录
mkdir -p "$SAVE_DIR"

# 运行分析
echo "模型路径: $MODEL_PATH"
echo "配置文件: $CONFIG_PATH"
echo "结果保存到: $SAVE_DIR"
echo ""

python "$SCRIPT_DIR/main_analysis.py" \
    --model_path "$MODEL_PATH" \
    --config "$CONFIG_PATH" \
    --save_dir "$SAVE_DIR" \
    --device "$DEVICE" \
    2>&1 | tee "$SAVE_DIR/analysis.log"

echo ""
echo "分析完成！结果保存在: $SAVE_DIR"
echo "日志文件: $SAVE_DIR/analysis.log"
