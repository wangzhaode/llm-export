# LLM-Export

[![PyPI version](https://badge.fury.io/py/llmexport.svg)](https://badge.fury.io/py/llmexport)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

[English](./README_en.md) | 中文

一个高效的大语言模型导出工具，能够将 LLM 模型导出为 ONNX 和 MNN 格式，支持量化优化和多模态模型。

## ✨ 主要特性

- 🚀 **动态形状支持**：优化原始代码，支持动态输入形状
- 🚀 **模型优化**：减少常量部分，提升推理性能
- 🚀 **自动优化**：集成 [OnnxSlim](https://github.com/inisis/OnnxSlim) 优化 ONNX 模型，性能提升约 5% (感谢 [@inisis](https://github.com/inisis))
- 🚀 **LoRA 支持**：支持 LoRA 权重的合并/分离导出
- 🚀 **量化技术**：支持 AWQ、GPTQ、HQQ等多种量化方法
- 🚀 **EAGLE 支持**：支持 EAGLE 推理加速技术
- 🚀 **多模态支持**：支持文本、图像、音频等多模态模型
- 🚀 **推理框架**：提供 [MNN](https://github.com/wangzhaode/mnn-llm) 和 [ONNX](https://github.com/wangzhaode/onnx-llm) 推理代码

## 📜 快速开始

### 安装

```bash
# 从 PyPI 安装（推荐）
pip install llmexport

# 从 GitHub 安装最新版本
pip install git+https://github.com/wangzhaode/llm-export@master

# 本地开发安装
git clone https://github.com/wangzhaode/llm-export
cd llm-export
pip install -e .
```

### 基本用法

#### 1. 下载模型

```bash
# 使用 Hugging Face CLI
huggingface-cli download Qwen/Qwen2.5-1.5B-Instruct --local-dir Qwen2.5-1.5B-Instruct

# 或使用 ModelScope（国内用户推荐）
modelscope download Qwen/Qwen2.5-1.5B-Instruct --local_dir Qwen2.5-1.5B-Instruct
```

#### 2. 模型测试

```bash
# 文本对话测试
llmexport --path Qwen2.5-1.5B-Instruct --test "你好，请介绍一下你自己"

# 多模态测试（图像+文本）
llmexport --path Qwen2-VL-2B-Instruct --test "<img>image_url</img>描述一下这张图片"
```

#### 3. 模型导出

```bash
# 导出为 ONNX 格式
llmexport --path Qwen2.5-1.5B-Instruct --export onnx

# 导出为 MNN 格式（默认 4bit 量化）
llmexport --path Qwen2.5-1.5B-Instruct --export mnn

# 自定义量化参数
llmexport --path Qwen2.5-1.5B-Instruct --export mnn --quant_bit 8 --quant_block 128

# 导出 EAGLE 模型
llmexport --path Qwen2.5-1.5B-Instruct --export mnn --eagle_path path/to/eagle
```

## 🔧 高级功能

### 模型导出选项

- **ONNX 导出**：使用 `--export onnx` 导出为 ONNX 格式
- **MNN 导出**：使用 `--export mnn` 导出为 MNN 格式
- **模型优化**：默认启用 OnnxSlim 优化，使用 `--onnx_slim` 显式启用
- **EAGLE 导出**：使用 `--eagle_path` 导出 EAGLE 加速模型

### 量化配置

- **量化位数**：`--quant_bit 4/8` （默认 4bit）
- **量化块大小**：`--quant_block 64/128` （默认 64）
- **LM Head 量化**：`--lm_quant_bit` 单独设置输出层量化
- **对称量化**：`--sym` 启用对称量化（无零点）

### 量化算法支持

- **AWQ 量化**：`--awq` 启用 AWQ 量化
- **HQQ 量化**：`--hqq` 启用 HQQ 量化
- **GPTQ 量化**：`--gptq_path` 加载 GPTQ 量化模型
- **Smooth 量化**：`--smooth` 启用 Smooth 量化

### LoRA 支持

- **LoRA 合并**：`--lora_path` 指定 LoRA 权重路径
- **LoRA 分离**：`--lora_split` 分离导出 LoRA 权重

### 多模态支持

- **视觉量化**：`--visual_quant_bit`、`--visual_quant_block` 设置视觉模块量化
- **视觉对称**：`--visual_sym` 视觉模块对称量化

### 其他选项

- **详细输出**：`--verbose` 显示详细日志
- **性能评估**：`--ppl` 获取所有 token 的 logits
- **自定义输出**：`--dst_path` 指定输出目录（默认 `./model`）
- **EAGLE 支持**：`--eagle_path` 指定 EAGLE 模型路径

## 📎 命令行参数

### 基本参数

| 参数 | 类型 | 说明 |
|------|------|------|
| `--path` | 必需 | 模型路径，支持本地目录或 Hugging Face 模型 ID |
| `--export` | 可选 | 导出格式：`onnx` 或 `mnn` |
| `--test` | 可选 | 测试查询字符串 |
| `--dst_path` | 可选 | 输出目录（默认 `./model`） |
| `--verbose` | 开关 | 显示详细日志 |

### 量化参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--quant_bit` | 4 | 量化位数（4 或 8） |
| `--quant_block` | 64 | 量化块大小（0 表示通道级量化） |
| `--lm_quant_bit` | 同 `quant_bit` | LM Head 层量化位数 |
| `--visual_quant_bit` | 模型相关 | 视觉模块量化位数 |
| `--visual_quant_block` | 模型相关 | 视觉模块量化块大小 |

### 量化算法

| 参数 | 说明 |
|------|------|
| `--awq` | 启用 AWQ 量化 |
| `--hqq` | 启用 HQQ 量化 |
| `--smooth` | 启用 Smooth 量化 |
| `--sym` | 启用对称量化（无零点） |
| `--visual_sym` | 视觉模块对称量化 |

### LoRA 支持

| 参数 | 说明 |
|------|------|
| `--lora_path` | LoRA 权重路径 |
| `--lora_split` | 分离导出 LoRA 权重 |

### EAGLE 支持

| 参数 | 说明 |
|------|------|
| `--eagle_path` | EAGLE 模型路径 |

### 其他选项

| 参数 | 说明 |
|------|------|
| `--tokenizer_path` | 分词器路径（默认使用 `--path`） |
| `--gptq_path` | GPTQ 量化模型路径 |
| `--mnnconvert` | 本地 MNNConvert 路径 |
| `--onnx_slim` | 启用 ONNX-Slim 优化 |
| `--ppl` | 获取所有 token 的 logits |
| `--seperate_embed` | 分离嵌入层以避免量化 |
| `--calib_data` | 校准数据路径 |

## 📋 支持模型

目前支持以下模型类型：

### 文本模型
- **Qwen 系列**：Qwen3、Qwen2.5、Qwen2、Qwen1.5、Qwen-VL 等
- **LLaMA 系列**：Llama-3.2、Llama-3、Llama-2 等
- **ChatGLM 系列**：ChatGLM4、ChatGLM3、ChatGLM2 等
- **Baichuan 系列**：Baichuan2-7B-Chat 等
- **Yi 系列**：Yi-6B-Chat 等
- **其他**：InternLM、DeepSeek、Phi、Gemma、TinyLlama、SmolLM 等

### 多模态模型
- **视觉模型**：Qwen2-VL、Qwen2.5-VL、Qwen3-VL、Llama-3.2-Vision、InternVL 等
- **音频模型**：Qwen2-Audio、Qwen2.5-Omni 等

### 嵌入模型
- **文本嵌入**：bge-large-zh、gte-multilingual 等

## 💾 模型下载

我们提供了已经优化的模型下载：

- **Hugging Face**：[taobao-mnn](https://huggingface.co/taobao-mnn)
- **ModelScope**：[MNN](https://modelscope.cn/organization/MNN)

部分热门模型：

| 模型 | Hugging Face | ModelScope |
|------|-------------|------------|
| DeepSeek-R1-1.5B-Qwen | [Q4_1](https://huggingface.co/taobao-mnn/DeepSeek-R1-1.5B-Qwen-MNN) | [Q4_1](https://modelscope.cn/models/MNN/DeepSeek-R1-1.5B-Qwen-MNN) |
| Qwen2.5-0.5B-Instruct | [Q4_1](https://huggingface.co/taobao-mnn/Qwen2.5-0.5B-Instruct-MNN) | [Q4_1](https://modelscope.cn/models/MNN/Qwen2.5-0.5B-Instruct-MNN) |
| Qwen2.5-1.5B-Instruct | [Q4_1](https://huggingface.co/taobao-mnn/Qwen2.5-1.5B-Instruct-MNN) | [Q4_1](https://modelscope.cn/models/MNN/Qwen2.5-1.5B-Instruct-MNN) |
| GPT-OSS-20B | [Q4_1](https://huggingface.co/taobao-mnn/gpt-oss-20b-MNN) | [Q4_1](https://modelscope.cn/models/MNN/gpt-oss-20b-MNN) |
| Qwen3-4B-Instruct-2507 | [Q4_1](https://huggingface.co/taobao-mnn/Qwen3-4B-Instruct-2507-MNN) | [Q4_1](https://modelscope.cn/models/MNN/Qwen3-4B-Instruct-2507-MNN) |

更多模型请查看完整列表。

## 🔗 相关项目

- **MNN 推理**：[mnn-llm](https://github.com/wangzhaode/mnn-llm) - MNN 框架的 LLM 推理库
- **ONNX 推理**：[onnx-llm](https://github.com/wangzhaode/onnx-llm)、[OnnxLLM](https://github.com/inisis/OnnxLLM) - ONNX 格式推理库
- **模型优化**：[OnnxSlim](https://github.com/inisis/OnnxSlim) - ONNX 模型优化工具

## 📄 许可证

本项目采用 [Apaache 2.0 许可证](./LICENSE)。