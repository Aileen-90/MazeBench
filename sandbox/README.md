# MazeBench Sandbox

## 模式
- **CLI**: 真人玩家交互 `python run_sandbox.py cli`
- **API**: REST服务 `python run_sandbox.py api`
- **AI**: AI模型测试 `python run_sandbox.py ai [maze]`

## AI测试
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
