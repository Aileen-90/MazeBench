# MazeBench Sandbox

## 坐标系统
所有坐标使用 (y, x) 格式，其中：
- y: 行号（垂直坐标，从上到下递增）
- x: 列号（水平坐标，从左到右递增）

## 模式
- **CLI**: 真人玩家交互 `python run_sandbox.py cli`
- **API**: REST服务 `python run_sandbox.py api`
- **AI**: AI模型测试 `python run_sandbox.py ai [maze]`

## AI测试
AI模型可以：
1. 输入单个动作：up/down/left/right
2. 输入路径坐标：如 (1,2),(1,3),(2,3) - 坐标格式为 (y,x)
   - 如果路径第一个坐标是当前自身坐标，系统会自动跳过

```bash
# 默认配置测试
python run_sandbox.py ai

# 指定迷宫
python run_sandbox.py ai maze_9x9_0

# 自定义参数
python run_sandbox.py ai maze_9x9_0 --model gpt-4 --max-steps 30
```

## 配置
```yaml
model: "gpt-4"  # AI模型
sandbox:
  enabled: true
  max_steps: 50
```

## 输出
结果保存到 `outputs/ai_sandbox_*.json`
