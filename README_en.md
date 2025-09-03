# LLM-Export

[![PyPI version](https://badge.fury.io/py/llmexport.svg)](https://badge.fury.io/py/llmexport)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

English | [中文](./README.md)

An efficient Large Language Model export tool that converts LLM models to ONNX and MNN formats, supporting quantization optimization and multimodal models.

## ✨ Key Features

- 🚀 **Dynamic Shape Support**: Optimized original code with dynamic input shape support
- 🚀 **Model Optimization**: Reduced constant parts for improved inference performance
- 🚀 **Automatic Optimization**: Integrated [OnnxSlim](https://github.com/inisis/OnnxSlim) for ONNX model optimization, ~5% performance improvement (Thanks [@inisis](https://github.com/inisis))
- 🚀 **LoRA Support**: Support for LoRA weight merging/splitting export
- 🚀 **Quantization Methods**: Support for AWQ, GPTQ, HQQ, and other quantization methods
- 🚀 **Multimodal Support**: Support for text, image, audio, and other multimodal models
- 🚀 **Inference Frameworks**: Provides [MNN](https://github.com/wangzhaode/mnn-llm) and [ONNX](https://github.com/wangzhaode/onnx-llm) inference code

## 📖 Quick Start

### Installation

```bash
# Install from PyPI (Recommended)
pip install llmexport

# Install latest version from GitHub
pip install git+https://github.com/wangzhaode/llm-export@master

# Local development installation
git clone https://github.com/wangzhaode/llm-export
cd llm-export
pip install -e .
```

### Basic Usage

#### 1. Download Model

```bash
# Using Hugging Face CLI
huggingface-cli download Qwen/Qwen2.5-1.5B-Instruct --local-dir Qwen2.5-1.5B-Instruct

# Or using ModelScope (Recommended for users in China)
modelscope download Qwen/Qwen2.5-1.5B-Instruct --local_dir Qwen2.5-1.5B-Instruct
```

#### 2. Model Testing

```bash
# Text conversation testing
llmexport --path Qwen2.5-1.5B-Instruct --test "Hello, please introduce yourself"

# Multimodal testing (Image + Text)
llmexport --path Qwen2-VL-2B-Instruct --test "<img>image_url</img>Describe this image"
```

#### 3. Model Export

```bash
# Export to ONNX format
llmexport --path Qwen2.5-1.5B-Instruct --export onnx

# Export to MNN format (Default 4bit quantization)
llmexport --path Qwen2.5-1.5B-Instruct --export mnn

# Custom quantization parameters
llmexport --path Qwen2.5-1.5B-Instruct --export mnn --quant_bit 8 --quant_block 128
```

## 🔧 Advanced Features

### Model Export Options

- **ONNX Export**: Use `--export onnx` to export to ONNX format
- **MNN Export**: Use `--export mnn` to export to MNN format
- **Model Optimization**: OnnxSlim optimization enabled by default, use `--onnx_slim` to explicitly enable

### Quantization Configuration

- **Quantization Bits**: `--quant_bit 4/8` (Default 4bit)
- **Quantization Block Size**: `--quant_block 64/128` (Default 64)
- **LM Head Quantization**: `--lm_quant_bit` separate setting for output layer quantization
- **Symmetric Quantization**: `--sym` enable symmetric quantization (no zero point)

### Quantization Algorithm Support

- **AWQ Quantization**: `--awq` enable AWQ quantization
- **HQQ Quantization**: `--hqq` enable HQQ quantization
- **GPTQ Quantization**: `--gptq_path` load GPTQ quantized model
- **Smooth Quantization**: `--smooth` enable Smooth quantization

### LoRA Support

- **LoRA Merging**: `--lora_path` specify LoRA weight path
- **LoRA Splitting**: `--lora_split` export LoRA weights separately

### Multimodal Support

- **Visual Quantization**: `--visual_quant_bit`, `--visual_quant_block` set visual module quantization
- **Visual Symmetric**: `--visual_sym` visual module symmetric quantization

### Other Options

- **Verbose Output**: `--verbose` show detailed logs
- **Performance Evaluation**: `--ppl` get logits for all tokens
- **Custom Output**: `--dst_path` specify output directory (default `./model`)

## 📎 Command Line Parameters

### Basic Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `--path` | Required | Model path, supports local directory or Hugging Face model ID |
| `--export` | Optional | Export format: `onnx` or `mnn` |
| `--test` | Optional | Test query string |
| `--dst_path` | Optional | Output directory (default `./model`) |
| `--verbose` | Flag | Show detailed logs |

### Quantization Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--quant_bit` | 4 | Quantization bits (4 or 8) |
| `--quant_block` | 64 | Quantization block size (0 means channel-wise) |
| `--lm_quant_bit` | Same as `quant_bit` | LM Head layer quantization bits |
| `--visual_quant_bit` | Model dependent | Visual module quantization bits |
| `--visual_quant_block` | Model dependent | Visual module quantization block size |

### Quantization Algorithms

| Parameter | Description |
|-----------|-------------|
| `--awq` | Enable AWQ quantization |
| `--hqq` | Enable HQQ quantization |
| `--smooth` | Enable Smooth quantization |
| `--sym` | Enable symmetric quantization (no zero point) |
| `--visual_sym` | Visual module symmetric quantization |

### LoRA Support

| Parameter | Description |
|-----------|-------------|
| `--lora_path` | LoRA weight path |
| `--lora_split` | Export LoRA weights separately |

### Other Options

| Parameter | Description |
|-----------|-------------|
| `--tokenizer_path` | Tokenizer path (default uses `--path`) |
| `--gptq_path` | GPTQ quantized model path |
| `--mnnconvert` | Local MNNConvert path |
| `--onnx_slim` | Enable ONNX-Slim optimization |
| `--ppl` | Get logits for all tokens |
| `--seperate_embed` | Separate embedding layer to avoid quantization |
| `--calib_data` | Calibration data path |

## Commad Args
```
usage: llmexport.py [-h] --path PATH [--type TYPE] [--tokenizer_path TOKENIZER_PATH] [--lora_path LORA_PATH] [--gptq_path GPTQ_PATH] [--dst_path DST_PATH]
                    [--verbose] [--test TEST] [--export EXPORT] [--onnx_slim] [--quant_bit QUANT_BIT] [--quant_block QUANT_BLOCK] [--lm_quant_bit LM_QUANT_BIT]
                    [--mnnconvert MNNCONVERT] [--ppl] [--awq] [--sym] [--tie_embed] [--lora_split]

llm_exporter

options:
  -h, --help            show this help message and exit
  --path PATH           path(`str` or `os.PathLike`):
                        Can be either:
                        	- A string, the *model id* of a pretrained model like `THUDM/chatglm-6b`. [TODO]
                        	- A path to a *directory* clone from repo like `../chatglm-6b`.
  --type TYPE           type(`str`, *optional*):
                        	The pretrain llm model type.
  --tokenizer_path TOKENIZER_PATH
                        tokenizer path, defaut is `None` mean using `--path` value.
  --lora_path LORA_PATH
                        lora path, defaut is `None` mean not apply lora.
  --gptq_path GPTQ_PATH
                        gptq path, defaut is `None` mean not apply gptq.
  --dst_path DST_PATH   export onnx/mnn model to path, defaut is `./model`.
  --verbose             Whether or not to print verbose.
  --test TEST           test model inference with query `TEST`.
  --export EXPORT       export model to an onnx/mnn model.
  --onnx_slim           Whether or not to use onnx-slim.
  --quant_bit QUANT_BIT
                        mnn quant bit, 4 or 8, default is 4.
  --quant_block QUANT_BLOCK
                        mnn quant block, default is 0 mean channle-wise.
  --lm_quant_bit LM_QUANT_BIT
                        mnn lm_head quant bit, 4 or 8, default is `quant_bit`.
  --mnnconvert MNNCONVERT
                        local mnnconvert path, if invalid, using pymnn.
  --ppl                 Whether or not to get all logits of input tokens.
  --awq                 Whether or not to use awq quant.
  --sym                 Whether or not to using symmetric quant (without zeropoint), defualt is False.
  --tie_embed           Whether or not to using tie_embedding, defualt is False.
  --lora_split          Whether or not export lora split, defualt is False.
```

## 📋 Supported Models

Currently supports the following model types:

### Text Models
- **Qwen Series**: Qwen2.5, Qwen2, Qwen1.5, Qwen-VL, etc.
- **LLaMA Series**: Llama-3.2, Llama-3, Llama-2, etc.
- **ChatGLM Series**: ChatGLM4, ChatGLM3, ChatGLM2, etc.
- **Baichuan Series**: Baichuan2-7B-Chat, etc.
- **Yi Series**: Yi-6B-Chat, etc.
- **Others**: InternLM, DeepSeek, Phi, Gemma, TinyLlama, etc.

### Multimodal Models
- **Vision Models**: Qwen2-VL, Qwen2.5-VL, Llama-3.2-Vision, InternVL, etc.
- **Audio Models**: Qwen2-Audio, Qwen2.5-Omni, etc.

### Embedding Models
- **Text Embedding**: bge-large-zh, gte-multilingual, etc.

## 💾 Model Downloads

We provide optimized model downloads:

- **Hugging Face**: [taobao-mnn](https://huggingface.co/taobao-mnn)
- **ModelScope**: [MNN](https://modelscope.cn/organization/MNN)

Popular models:

| Model | Hugging Face | ModelScope |
|-------|-------------|------------|
| DeepSeek-R1-1.5B-Qwen | [Q4_1](https://huggingface.co/taobao-mnn/DeepSeek-R1-1.5B-Qwen-MNN) | [Q4_1](https://modelscope.cn/models/MNN/DeepSeek-R1-1.5B-Qwen-MNN) |
| Qwen2.5-0.5B-Instruct | [Q4_1](https://huggingface.co/taobao-mnn/Qwen2.5-0.5B-Instruct-MNN) | [Q4_1](https://modelscope.cn/models/MNN/Qwen2.5-0.5B-Instruct-MNN) |
| Qwen2.5-1.5B-Instruct | [Q4_1](https://huggingface.co/taobao-mnn/Qwen2.5-1.5B-Instruct-MNN) | [Q4_1](https://modelscope.cn/models/MNN/Qwen2.5-1.5B-Instruct-MNN) |
| GPT-OSS-20B | [Q4_1](https://huggingface.co/taobao-mnn/gpt-oss-20b-MNN) | [Q4_1](https://modelscope.cn/models/MNN/gpt-oss-20b-MNN) |
| Qwen3-4B-Instruct-2507 | [Q4_1](https://huggingface.co/taobao-mnn/Qwen3-4B-Instruct-2507-MNN) | [Q4_1](https://modelscope.cn/models/MNN/Qwen3-4B-Instruct-2507-MNN) |

See the complete list for more models.

## 🔗 Related Projects

- **MNN Inference**: [mnn-llm](https://github.com/wangzhaode/mnn-llm) - LLM inference library for MNN framework
- **ONNX Inference**: [onnx-llm](https://github.com/wangzhaode/onnx-llm), [OnnxLLM](https://github.com/inisis/OnnxLLM) - ONNX format inference libraries
- **Model Optimization**: [OnnxSlim](https://github.com/inisis/OnnxSlim) - ONNX model optimization tool

## 📄 License

This project is licensed under the [MIT License](https://opensource.org/licenses/MIT).1.7B-Instruct-MNN) |