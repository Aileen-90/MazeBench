# 使用Llama模型运行MazeBenchmark

本指南将介绍如何在MazeBenchmark项目中使用Llama模型进行迷宫测试。

## 1. 准备工作

### 1.1 安装Ollama

Llama模型通常通过Ollama服务提供，首先需要安装Ollama：

```bash
# Linux/macOS
curl -fsSL https://ollama.com/install.sh | sh

# Windows
# 访问 https://ollama.com/download 下载安装程序
```

### 1.2 启动Ollama服务

```bash
# Linux/macOS
ollama serve

# Windows
# Ollama安装后会自动启动服务
```

### 1.3 拉取Llama模型

```bash
# 拉取Llama 3.1 模型
ollama pull llama3.1

# 或者拉取其他Llama模型
ollama pull llama3.1:8b-instruct
ollama pull llama3.1:70b-instruct
```

## 2. 配置MazeBenchmark

### 2.1 使用提供的Llama配置

我们已经创建了专门用于Llama测试的配置文件。现在支持两种方式运行Llama模型：

#### 2.1.1 使用Ollama适配器（推荐）

配置文件 `config/llama_config.yaml` 使用新的Ollama适配器：

```yaml
# Llama模型测试配置 - Ollama适配器
PROVIDER: "ollama"
model: "llama3.1:70b-instruct"  # Llama模型名称
base_url: "http://localhost:11434/v1"  # Ollama服务URL

temperature: 0.0
mode: "text2d"
output_dir: "outputs_llama/"
mazes_path: "mazes/"

sandbox:
  enabled: true
  max_steps: 50
  memory: 10
  visibility: -1
```

#### 2.1.2 使用OpenAI兼容模式（旧方式，兼容）

如果需要兼容旧的配置方式，可以继续使用：

```yaml
# Llama模型测试配置 - OpenAI兼容模式
PROVIDER: "openai"
model: "llama3.1:70b-instruct"  # Llama模型名称
OPENAI_API_KEY: "ollama"  # Ollama服务不需要API密钥，使用任意值即可
OPENAI_API_BASE: "http://localhost:11434/v1"  # Ollama服务URL

temperature: 0.0
mode: "text2d"
output_dir: "outputs_llama/"
mazes_path: "mazes/"

sandbox:
  enabled: true
  max_steps: 50
  memory: 10
  visibility: -1
```

### 2.2 调整模型名称

根据您拉取的实际模型，调整配置文件中的 `model` 字段：

```yaml
# Llama 3.1 8B模型
model: "llama3.1:8b-instruct"

# Llama 3.1 70B模型
model: "llama3.1:70b-instruct"
```

## 3. 运行测试

### 3.1 测试Llama模型连接

```bash
python3 test_llama.py
```

### 3.2 运行单个Llama模型测试

```bash
# 使用主程序运行测试
python3 main.py --config config/llama_config.yaml

# 或者使用沙盒模式
python3 run_sandbox.py --mode ai --config config/llama_config.yaml
```

### 3.3 运行多模型测试（包括Llama）

修改 `multi_model_test/run_multi_test.sh` 文件，添加Llama模型：

```bash
# 在MODELS数组中添加Llama模型
MODELS=("llama3.1:8b-instruct" "gpt-4" "qwen3-max-preview")

# 设置正确的API配置
export OPENAI_API_KEY="ollama"
export OPENAI_API_BASE="http://localhost:11434/v1"

# 运行多模型测试
./run_multi_test.sh
```

## 4. 常见问题排查

### 4.1 连接错误

如果遇到连接错误：

```
Connection error.
```

请检查：
- Ollama服务是否正在运行（`ollama serve`）
- 配置文件中的 `base_url`（Ollama适配器）或 `OPENAI_API_BASE`（OpenAI兼容模式）是否正确（默认：http://localhost:11434/v1）
- 网络是否有防火墙限制

### 4.2 模型未找到

如果遇到模型未找到错误：

```
Model not found.
```

请检查：
- 是否已使用 `ollama pull` 拉取了指定的模型
- 配置文件中的模型名称是否与拉取的模型完全匹配

### 4.3 API密钥错误

- **Ollama适配器**：不需要API密钥，配置中无需设置
- **OpenAI兼容模式**：不需要真实的API密钥，可以使用任意值：

```yaml
OPENAI_API_KEY: "ollama"  # 任意值都可以
```

## 5. 性能优化建议

1. **选择合适的模型大小**：
   - 8B模型：适合快速测试
   - 70B模型：适合更复杂的迷宫任务

2. **调整沙盒配置**：
   
```yaml
sandbox:
  max_steps: 100  # 增加最大步数
  memory: 20      # 增加记忆长度
```

3. **使用GPU加速**：
   确保Ollama服务可以访问GPU资源，以提高模型推理速度。

## 6. 示例命令

```bash
# 测试5x5迷宫（使用Ollama适配器）
python3 run_sandbox.py --mode ai --config config/llama_config.yaml --maze mazes/maze_5x5_0.json --model llama3.1:8b-instruct

# 运行10x10迷宫测试（使用Ollama适配器）
python3 main.py --config config/llama_config.yaml --size 10x10 --count 3

# 运行10x10迷宫测试（使用OpenAI兼容模式）
python3 main.py --config config/llama_openai_config.yaml --size 10x10 --count 3
```

## 7. 查看结果

测试结果将保存在 `outputs_llama/` 目录下，包括：
- 迷宫测试结果JSON文件
- 详细的运行日志
- 可视化报告（如果启用）

祝您使用Llama模型进行迷宫测试愉快！