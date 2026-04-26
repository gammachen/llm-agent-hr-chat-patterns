# vLLM 文本生成与问答应用

这个项目使用vLLM和量化的facebook/opt-125m模型实现了文本生成与问答功能，并提供了Streamlit和Gradio两种界面。

## 功能特点

- 使用vLLM高效推理引擎
- 采用量化的facebook/opt-125m模型
- 支持文本生成和问答两种模式
- 提供Streamlit Web界面和Gradio API界面

## 安装依赖

```bash
pip install torch vllm streamlit gradio
```

## 使用方法

### 运行Streamlit界面

```bash
python vllm_streamlit_app.py
```

或者直接使用streamlit命令：

```bash
streamlit run vllm_streamlit_app.py
```

### 运行Gradio界面

```bash
python vllm_gradio_app.py
```

### 使用主程序（支持选择界面类型）

```bash
# 运行Streamlit界面（默认）
python test_vllm.py

# 运行Gradio界面
python test_vllm.py --app gradio
```

## 配置选项

两种界面都提供以下配置选项：

- **最大生成长度**：控制生成文本的最大长度（16-512）
- **温度**：控制生成文本的随机性（0.0-1.0）
- **Top-p**：控制生成文本的多样性（0.0-1.0）

## 模型说明

本项目使用的是facebook/opt-125m模型，这是一个相对较小的语言模型，适合在资源有限的环境中运行。模型使用float16进行量化，以提高推理效率。

## 注意事项

- 首次运行时，系统会自动下载模型，这可能需要一些时间
- 模型运行需要GPU支持，请确保您的系统有可用的GPU资源
- 如果遇到内存不足的问题，可以尝试调整`gpu_memory_utilization`参数