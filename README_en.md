# llm-export

[中文](./README_en.md)

llm-export is a tool for exporting llm models, capable of converting llm models into ONNX or MNN models.
- 🚀 Optimized the original code to support dynamic shapes
- 🚀 Optimized the original code to reduce the constant portion
- 🚀 Using [OnnxSlim](https://github.com/inisis/OnnxSlim) slim onnx model，speed up 5%; by [@inisis](https://github.com/inisis)
- 🚀 Support export lora weight to onnx or MNN model
- 🚀 MNN inference code[mnn-llm](https://github.com/wangzhaode/mnn-llm)
- 🚀 Onnx inference code [onnx-llm](https://github.com/wangzhaode/onnx-llm), [OnnxLLM](https://github.com/inisis/OnnxLLM)

## Install

```sh
# pip install
pip install llmexport

# git install
pip install git+https://github.com/wangzhaode/llm-export@master

# local install
git clone https://github.com/wangzhaode/llm-export && cd llm-export/
pip install .
```

## Usage
1. download the model, Clone the LLM project that you want to export locally, such as: chatglm2-6b
```sh
git clone https://huggingface.co/Qwen/Qwen2-1.5B-Instruct
# If downloading from Hugging Face is slow, you can use ModelScope
git clone https://modelscope.cn/qwen/Qwen2-1.5B-Instruct.git
```
2. test the model
```sh
# Test text
llmexport --path Qwen2-1.5B-Instruct --test "Hello"
# Test image text
llmexport --path Qwen2-VL-2B-Instruct  --test "<img>https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg</img>Describe the content of the picture"
```
2. export the model
```sh
# export chatglm2-6b to onnx
llmexport --path Qwen2-1.5B-Instruct --export onnx
# export chatglm2-6b to mnn and quant
llmexport --path Qwen2-1.5B-Instruct --export mnn --quant_bit 4 --quant_block 128
```

## Features
- Supports exporting the entire model as a onnx model or mnn model, use `--export onnx/mnn`
- Default using onnx-slim, skip using `--skip_slim`
- Support merge lora.

## Commad Args
```
usage: llmexport [-h] --path PATH [--type TYPE] [--lora_path LORA_PATH] [--dst_path DST_PATH] [--test TEST] [--export EXPORT] [--skip_slim] [--quant_bit QUANT_BIT] [--quant_block QUANT_BLOCK]
                 [--lm_quant_bit LM_QUANT_BIT]

llm_exporter

optional arguments:
  -h, --help            show this help message and exit
  --path PATH           path(`str` or `os.PathLike`):
                        Can be either:
                        	- A string, the *model id* of a pretrained model like `THUDM/chatglm-6b`. [TODO]
                        	- A path to a *directory* clone from repo like `../chatglm-6b`.
  --type TYPE           type(`str`, *optional*):
                        	The pretrain llm model type.
  --lora_path LORA_PATH
                        lora path, defaut is `None` mean not apply lora.
  --dst_path DST_PATH   export onnx/mnn model to path, defaut is `./model`.
  --test TEST           test model inference with query `TEST`.
  --export EXPORT       export model to an onnx/mnn model.
  --skip_slim           Whether or not to skip onnx-slim.
  --quant_bit QUANT_BIT
                        mnn quant bit, 4 or 8, default is 4.
  --quant_block QUANT_BLOCK
                        mnn quant block, default is 0 mean channle-wise.
  --lm_quant_bit LM_QUANT_BIT
                        mnn lm_head quant bit, 4 or 8, default is `quant_bit`.
```

## Model Download

|   Model  | ModelScope  | Hugging Face |
| -------- | ----------- | ------------ |
| [Qwen-VL-Chat](https://modelscope.cn/models/qwen/Qwen-VL-Chat/summary) | [Q4_1](https://modelscope.cn/models/MNN/Qwen-VL-Chat-MNN) | [Q4_1](https://huggingface.co/alibaba-mnn/Qwen-VL-Chat-MNN) |
| [Baichuan2-7B-Chat](https://modelscope.cn/models/baichuan-inc/Baichuan2-7B-Chat/summary) | [Q4_1](https://modelscope.cn/models/MNN/Baichuan2-7B-Chat-MNN) | [Q4_1](https://huggingface.co/alibaba-mnn/Baichuan2-7B-Chat-MNN) |
| [bge-large-zh](https://modelscope.cn/models/AI-ModelScope/bge-large-zh/summary) | [Q4_1](https://modelscope.cn/models/MNN/bge-large-zh-MNN) | [Q4_1](https://huggingface.co/alibaba-mnn/bge-large-zh-MNN) |
| [chatglm-6b](https://modelscope.cn/models/ZhipuAI/ChatGLM-6B/summary) | [Q4_1](https://modelscope.cn/models/MNN/chatglm-6b-MNN) | [Q4_1](https://huggingface.co/alibaba-mnn/chatglm-6b-MNN) |
| [chatglm2-6b](https://modelscope.cn/models/ZhipuAI/chatglm2-6b/summary) | [Q4_1](https://modelscope.cn/models/MNN/chatglm2-6b-MNN) | [Q4_1](https://huggingface.co/alibaba-mnn/chatglm2-6b-MNN) |
| [chatglm3-6b](https://modelscope.cn/models/ZhipuAI/chatglm3-6b/summary) | [Q4_1](https://modelscope.cn/models/MNN/chatglm3-6b-MNN) | [Q4_1](https://huggingface.co/alibaba-mnn/chatglm3-6b-MNN) |
| [codegeex2-6b](https://modelscope.cn/models/MNN/codegeex2-6b-MNN/summary) | [Q4_1](https://modelscope.cn/models/MNN/codegeex2-6b-MNN) | [Q4_1](https://huggingface.co/alibaba-mnn/codegeex2-6b-MNN) |
| [deepseek-llm-7b-chat](https://modelscope.cn/models/deepseek-ai/deepseek-llm-7b-chat/summary) | [Q4_1](https://modelscope.cn/models/MNN/deepseek-llm-7b-chat-MNN) | [Q4_1](https://huggingface.co/alibaba-mnn/deepseek-llm-7b-chat-MNN) |
| [gemma-2-2b-it](https://modelscope.cn/models/llm-research/gemma-2-2b-it) | [Q4_1](https://modelscope.cn/models/MNN/gemma-2-2b-it-MNN) | [Q4_1](https://huggingface.co/alibaba-mnn/gemma-2-2b-it-MNN) |
| [glm-4-9b-chat](https://modelscope.cn/models/ZhipuAI/glm-4-9b-chat/summary) | [Q4_1](https://modelscope.cn/models/MNN/glm-4-9b-chat-MNN) | [Q4_1](https://huggingface.co/alibaba-mnn/glm-4-9b-chat-MNN) |
| [gte_sentence-embedding_multilingual-base](https://modelscope.cn/models/iic/gte_sentence-embedding_multilingual-base/summary) | [Q4_1](https://modelscope.cn/models/MNN/gte_sentence-embedding_multilingual-base-MNN) | [Q4_1](https://huggingface.co/alibaba-mnn/gte_sentence-embedding_multilingual-base-MNN) |
| [internlm-chat-7b](https://modelscope.cn/models/AI-ModelScope/internlm-chat-7b/summary) | [Q4_1](https://modelscope.cn/models/MNN/internlm-chat-7b-MNN) | [Q4_1](https://huggingface.co/alibaba-mnn/internlm-chat-7b-MNN) |
| [Llama-2-7b-chat](https://modelscope.cn/models/modelscope/Llama-2-7b-chat-ms/summary) | [Q4_1](https://modelscope.cn/models/MNN/Llama-2-7b-chat-MNN) | [Q4_1](https://huggingface.co/alibaba-mnn/Llama-2-7b-chat-MNN) |
| [Llama-3-8B-Instruct](https://modelscope.cn/models/modelscope/Meta-Llama-3-8B-Instruct/summary) | [Q4_1](https://modelscope.cn/models/MNN/Llama-3-8B-Instruct-MNN) | [Q4_1](https://huggingface.co/alibaba-mnn/Llama-3-8B-Instruct-MNN) |
| [Llama-3.2-1B-Instruct](https://modelscope.cn/models/LLM-Research/Llama-3.2-1B-Instruct/summary) | [Q4_1](https://modelscope.cn/models/MNN/Llama-3.2-1B-Instruct-MNN) | [Q4_1](https://huggingface.co/alibaba-mnn/Llama-3.2-1B-Instruct-MNN) |
| [Llama-3.2-3B-Instruct](https://modelscope.cn/models/LLM-Research/Llama-3.2-3B-Instruct/summary) | [Q4_1](https://modelscope.cn/models/MNN/Llama-3.2-3B-Instruct-MNN) | [Q4_1](https://huggingface.co/alibaba-mnn/Llama-3.2-3B-Instruct-MNN) |
| [OpenELM-1_1B-Instruct](https://huggingface.co/apple/OpenELM-1_1B-Instruct) | [Q4_1](https://modelscope.cn/models/MNN/OpenELM-1_1B-Instruct-MNN) | [Q4_1](https://huggingface.co/alibaba-mnn/OpenELM-1_1B-Instruct-MNN) |
| [OpenELM-270M-Instruct](https://huggingface.co/apple/OpenELM-270M-Instruct) | [Q4_1](https://modelscope.cn/models/MNN/OpenELM-270M-Instruct-MNN) | [Q4_1](https://huggingface.co/alibaba-mnn/OpenELM-270M-Instruct-MNN) |
| [OpenELM-3B-Instruct](https://huggingface.co/apple/OpenELM-3B-Instruct) | [Q8_1](https://modelscope.cn/models/MNN/OpenELM-3B-Instruct-MNN) | [Q8_1](https://huggingface.co/alibaba-mnn/OpenELM-3B-Instruct-MNN) |
| [OpenELM-450M-Instruct](https://huggingface.co/apple/OpenELM-450M-Instruct) | [Q4_1](https://modelscope.cn/models/MNN/OpenELM-450M-Instruct-MNN) | [Q4_1](https://huggingface.co/alibaba-mnn/OpenELM-450M-Instruct-MNN) |
| [phi-2](https://modelscope.cn/models/mengzhao/phi-2/summary) | [Q4_1](https://modelscope.cn/models/MNN/phi-2-MNN) | [Q4_1](https://huggingface.co/alibaba-mnn/phi-2-MNN) |
| [qwen/Qwen-1_8B-Chat](https://modelscope.cn/models/qwen/Qwen-1_8B-Chat/summary) | [Q4_1](https://modelscope.cn/models/MNN/Qwen-1_8B-Chat-MNN) | [Q4_1](https://huggingface.co/alibaba-mnn/Qwen-1_8B-Chat-MNN) |
| [Qwen-7B-Chat](https://modelscope.cn/models/qwen/Qwen-7B-Chat/summary) | [Q4_1](https://modelscope.cn/models/MNN/Qwen-7B-Chat-MNN) | [Q4_1](https://huggingface.co/alibaba-mnn/Qwen-7B-Chat-MNN) |
| [Qwen1.5-0.5B-Chat](https://modelscope.cn/models/qwen/Qwen1.5-0.5B-Chat/summary) | [Q4_1](https://modelscope.cn/models/MNN/Qwen1.5-0.5B-Chat-MNN) | [Q4_1](https://huggingface.co/alibaba-mnn/Qwen1.5-0.5B-Chat-MNN) |
| [Qwen1.5-1.8B-Chat](https://modelscope.cn/models/qwen/Qwen1.5-1.8B-Chat/summary) | [Q4_1](https://modelscope.cn/models/MNN/Qwen1.5-1.8B-Chat-MNN) | [Q4_1](https://huggingface.co/alibaba-mnn/Qwen1.5-1.8B-Chat-MNN) |
| [Qwen1.5-4B-Chat](https://modelscope.cn/models/qwen/Qwen1.5-4B-Chat/summary) | [Q4_1](https://modelscope.cn/models/MNN/Qwen1.5-4B-Chat-MNN) | [Q4_1](https://huggingface.co/alibaba-mnn/Qwen1.5-4B-Chat-MNN) |
| [Qwen1.5-7B-Chat](https://modelscope.cn/models/qwen/Qwen1.5-7B-Chat/summary) | [Q4_1](https://modelscope.cn/models/MNN/Qwen1.5-7B-Chat-MNN) | [Q4_1](https://huggingface.co/alibaba-mnn/Qwen1.5-7B-Chat-MNN) |
| [Qwen2-0.5B-Instruct](https://modelscope.cn/models/qwen/Qwen2-0.5B-Instruct/summary) | [Q4_1](https://modelscope.cn/models/MNN/Qwen2-0.5B-Instruct-MNN) | [Q4_1](https://huggingface.co/alibaba-mnn/Qwen2-0.5B-Instruct-MNN) |
| [Qwen2-1.5B-Instruct](https://modelscope.cn/models/qwen/Qwen2-1.5B-Instruct/summary) | [Q4_1](https://modelscope.cn/models/MNN/Qwen2-1.5B-Instruct-MNN) | [Q4_1](https://huggingface.co/alibaba-mnn/Qwen2-1.5B-Instruct-MNN) |
| [Qwen2-7B-Instruct](https://modelscope.cn/models/qwen/Qwen2-7B-Instruct/summary) | [Q4_1](https://modelscope.cn/models/MNN/Qwen2-7B-Instruct-MNN) | [Q4_1](https://huggingface.co/alibaba-mnn/Qwen2-7B-Instruct-MNN) |
| [Qwen2-VL-2B-Instruct](https://modelscope.cn/models/qwen/Qwen2-VL-2B-Instruct/summary) | [Q4_1](https://modelscope.cn/models/MNN/Qwen2-VL-2B-Instruct-MNN) | [Q4_1](https://huggingface.co/alibaba-mnn/Qwen2-VL-2B-Instruct-MNN) |
| [Qwen2-VL-7B-Instruct](https://modelscope.cn/models/qwen/Qwen2-VL-7B-Instruct/summary) | [Q4_1](https://modelscope.cn/models/MNN/Qwen2-VL-7B-Instruct-MNN) | [Q4_1](https://huggingface.co/alibaba-mnn/Qwen2-VL-7B-Instruct-MNN) |
| [Qwen2.5-0.5B-Instruct](https://modelscope.cn/models/qwen/Qwen2.5-0.5B-Instruct/summary) | [Q4_1](https://modelscope.cn/models/MNN/Qwen2.5-0.5B-Instruct-MNN) | [Q4_1](https://huggingface.co/alibaba-mnn/Qwen2.5-0.5B-Instruct-MNN) |
| [Qwen2.5-1.5B-Instruct](https://modelscope.cn/models/qwen/Qwen2.5-1.5B-Instruct/summary) | [Q4_1](https://modelscope.cn/models/MNN/Qwen2.5-1.5B-Instruct-MNN) | [Q4_1](https://huggingface.co/alibaba-mnn/Qwen2.5-1.5B-Instruct-MNN) |
| [Qwen2.5-3B-Instruct](https://modelscope.cn/models/qwen/Qwen2.5-3B-Instruct/summary) | [Q4_1](https://modelscope.cn/models/MNN/Qwen2.5-3B-Instruct-MNN) | [Q4_1](https://huggingface.co/alibaba-mnn/Qwen2.5-3B-Instruct-MNN) |
| [Qwen2.5-7B-Instruct](https://modelscope.cn/models/qwen/Qwen2.5-7B-Instruct/summary) | [Q4_1](https://modelscope.cn/models/MNN/Qwen2.5-7B-Instruct-MNN) | [Q4_1](https://huggingface.co/alibaba-mnn/Qwen2.5-7B-Instruct-MNN) |
| [Qwen2.5-Coder-1.5B-Instruct](https://modelscope.cn/models/qwen/Qwen2.5-Coder-1.5B-Instruct/summary) | [Q4_1](https://modelscope.cn/models/MNN/Qwen2.5-Coder-1.5B-Instruct-MNN) | [Q4_1](https://huggingface.co/alibaba-mnn/Qwen2.5-Coder-1.5B-Instruct-MNN) |
| [Qwen2.5-Coder-7B-Instruct](https://modelscope.cn/models/qwen/Qwen2.5-Coder-7B-Instruct/summary) | [Q4_1](https://modelscope.cn/models/MNN/Qwen2.5-Coder-7B-Instruct-MNN) | [Q4_1](https://huggingface.co/alibaba-mnn/Qwen2.5-Coder-7B-Instruct-MNN) |
| [Qwen2.5-Math-1.5B-Instruct](https://modelscope.cn/models/qwen/Qwen2.5-Math-1.5B-Instruct/summary) | [Q4_1](https://modelscope.cn/models/MNN/Qwen2.5-Math-1.5B-Instruct-MNN) | [Q4_1](https://huggingface.co/alibaba-mnn/Qwen2.5-Math-1.5B-Instruct-MNN) |
| [Qwen2.5-Math-7B-Instruct](https://modelscope.cn/models/qwen/Qwen2.5-Math-7B-Instruct/summary) | [Q4_1](https://modelscope.cn/models/MNN/Qwen2.5-Math-7B-Instruct-MNN) | [Q4_1](https://huggingface.co/alibaba-mnn/Qwen2.5-Math-7B-Instruct-MNN) |
| [reader-lm-0.5b](https://huggingface.co/jinaai/reader-lm-0.5b) | [Q4_1](https://modelscope.cn/models/MNN/reader-lm-0.5b-MNN) | [Q4_1](https://huggingface.co/alibaba-mnn/reader-lm-0.5b-MNN) |
| [reader-lm-1.5b](https://huggingface.co/jinaai/reader-lm-1.5b) | [Q4_1](https://modelscope.cn/models/MNN/reader-lm-1.5b-MNN) | [Q4_1](https://huggingface.co/alibaba-mnn/reader-lm-1.5b-MNN) |
| [TinyLlama-1.1B-Chat-v1.0](https://modelscope.cn/models/AI-ModelScope/TinyLlama-1.1B-Chat-v1.0/summary) | [Q4_1](https://modelscope.cn/models/MNN/TinyLlama-1.1B-Chat-MNN) | [Q4_1](https://huggingface.co/alibaba-mnn/TinyLlama-1.1B-Chat-MNN) |
| [Yi-6B-Chat](https://modelscope.cn/models/01ai/Yi-6B-Chat/summary) | [Q4_1](https://modelscope.cn/models/MNN/Yi-6B-Chat-MNN) | [Q4_1](https://huggingface.co/alibaba-mnn/Yi-6B-Chat-MNN) |
| [MobileLLM-125M](https://huggingface.co/facebook/MobileLLM-125M) | [Q4_1](https://modelscope.cn/models/MNN/MobileLLM-125M-MNN) | [Q4_1](https://huggingface.co/alibaba-mnn/MobileLLM-125M-MNN) |
| [MobileLLM-350M](https://huggingface.co/facebook/MobileLLM-350M) | [Q4_1](https://modelscope.cn/models/MNN/MobileLLM-350M-MNN) | [Q4_1](https://huggingface.co/alibaba-mnn/MobileLLM-350M-MNN) |
| [MobileLLM-600M](https://huggingface.co/facebook/MobileLLM-600M) | [Q4_1](https://modelscope.cn/models/MNN/MobileLLM-600M-MNN) | [Q4_1](https://huggingface.co/alibaba-mnn/MobileLLM-600M-MNN) |
| [MobileLLM-1B](https://huggingface.co/facebook/MobileLLM-1B) | [Q4_1](https://modelscope.cn/models/MNN/MobileLLM-1B-MNN) | [Q4_1](https://huggingface.co/alibaba-mnn/MobileLLM-1B-MNN) |