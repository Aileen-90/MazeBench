#!/bin/bash
# MazeBench 多模型迷宫测试运行脚本

echo "MazeBench 多模型迷宫测试"
echo "=========================="

# 设置默认参数
MODELS="${MODELS:-llama3:8b}"
SIZES="${SIZES:-5x5}"
TRIALS="${TRIALS:-5}"
WORKERS="${WORKERS:-10}"
OUTPUT_DIR="${OUTPUT_DIR:-multi_model_results}"

# 转换逗号分隔的字符串为数组
IFS=',' read -ra MODEL_ARRAY <<< "$MODELS"
IFS=',' read -ra SIZE_ARRAY <<< "$SIZES"

echo "测试配置:"
echo "  模型: ${MODEL_ARRAY[*]}"
echo "  迷宫大小: ${SIZE_ARRAY[*]}"
echo "  每迷宫测试次数: $TRIALS"
echo "  并发线程数: $WORKERS"
echo "  输出目录: $OUTPUT_DIR"
echo ""

# 构建命令 (从项目根目录运行)
CMD="python multi_model_test/multi_model_maze_test.py"
for model in "${MODEL_ARRAY[@]}"; do
    CMD="$CMD --models $model"
done

for size in "${SIZE_ARRAY[@]}"; do
    CMD="$CMD --sizes $size"
done

CMD="$CMD --trials $TRIALS --workers $WORKERS --output-dir $OUTPUT_DIR"

echo "执行命令: $CMD"
echo ""

# 执行测试
eval $CMD