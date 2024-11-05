# llm-export

[English](./README_en.md)

llm-export是一个llm模型导出工具，能够将llm模型导出为onnx和mnn模型。

- 🚀 优化原始代码，支持动态形状
- 🚀 优化原始代码，减少常量部分
- 🚀 使用[OnnxSlim](https://github.com/inisis/OnnxSlim)优化onnx模型，性能提升约5%; by [@inisis](https://github.com/inisis)
- 🚀 支持将lora权重导出为onnx和mnn
- 🚀 MNN推理代码[mnn-llm](https://github.com/wangzhaode/mnn-llm)
- 🚀 Onnx推理代码[onnx-llm](https://github.com/wangzhaode/onnx-llm), [OnnxLLM](https://github.com/inisis/OnnxLLM)

## 安装
```sh
# pip install
pip install llmexport

# git install
pip install git+https://github.com/wangzhaode/llm-export@master

# local install
git clone https://github.com/wangzhaode/llm-export && cd llm-export/
pip install .
```

## 用法

1. 下载模型
```sh
git clone https://huggingface.co/Qwen/Qwen2-1.5B-Instruct
# 如果huggingface下载慢可以使用modelscope
git clone https://modelscope.cn/qwen/Qwen2-1.5B-Instruct.git
```
2. 测试模型
```sh
# 测试文本输入
llmexport --path Qwen2-1.5B-Instruct --test "你好"
# 测试图像文本
llmexport --path Qwen2-VL-2B-Instruct  --test "<img>https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg</img>介绍一下图片里的内容"
```

3. 导出模型
```sh
# 将Qwen2-1.5B-Instruct导出为onnx模型
llmexport --path Qwen2-1.5B-Instruct --export onnx
# 将Qwen2-1.5B-Instruct导出为mnn模型, 量化参数为4bit, blokc-wise = 128
llmexport --path Qwen2-1.5B-Instruct --export mnn --quant_bit 4 --quant_block 128
```

## 功能
- 支持将模型为onnx或mnn模型，使用`--export onnx`或`--export mnn`
- 支持对模型进行对话测试，使用`--test $query`会返回llm的回复内容
- 默认会使用onnx-slim对onnx模型进行优化，跳过该步骤使用`--skip_slim`
- 支持合并lora权重后导出，指定lora权重的目录使用`--lora_path`
- 制定量化bit数使用`--quant_bit`；量化的block大小使用`--quant_block`
- 使用`--lm_quant_bit`来制定lm_head层权重的量化bit数，不指定则使用`--quant_bit`的量化bit数
- 支持使用自己编译的`MNNConvert`，使用`--mnnconvert`

## 参数
```
usage: llmexport.py [-h] --path PATH [--type TYPE] [--lora_path LORA_PATH] [--dst_path DST_PATH] [--test TEST] [--export EXPORT]
                    [--skip_slim] [--quant_bit QUANT_BIT] [--quant_block QUANT_BLOCK] [--lm_quant_bit LM_QUANT_BIT]
                    [--mnnconvert MNNCONVERT]

llm_exporter

options:
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
  --mnnconvert MNNCONVERT
                        local mnnconvert path, if invalid, using pymnn.
```

## 模型下载

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
