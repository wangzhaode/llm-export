# llm-export

[中文](./README_en.md)

llm-export is a tool for exporting llm models, capable of converting llm models into ONNX or MNN models.
- 🚀 Optimized the original code to support dynamic shapes
- 🚀 Optimized the original code to reduce the constant portion
- 🚀 Using [OnnxSlim](https://github.com/inisis/OnnxSlim) slim onnx model，speed up 5%; by [@inisis](https://github.com/inisis)
- 🚀 Support export lora weight to onnx or MNN model
- 🚀 Onnx inference code [OnnxLLM](https://github.com/inisis/OnnxLLM)

## Model Support and Downloads

## Usage
1. Clone this project locally
```sh
git clnoe git@github.com:wangzhaode/llm-export.git
```
2. Clone the LLM project that you want to export locally, such as: chatglm2-6b
```sh
git clone https://huggingface.co/THUDM/chatglm2-6b
# If downloading from Hugging Face is slow, you can use ModelScope
git clone https://modelscope.cn/ZhipuAI/chatglm2-6b.git
```
3. export the model
```sh
cd mnn-llm
cd mnn-llm
# export chatglm2-6b to onnx
python llm_export.py --path ../chatglm2-6b --export onnx
# export chatglm2-6b to mnn and quant
python llm_export.py --path ../chatglm2-6b --export mnn --quant_bit 4 --quant_block 128
```

## Features
- Supports exporting the entire model as a onnx model or mnn model, use `--export onnx/mnn`
- Default using onnx-slim, skip using `--skip_slim`
- Support merge lora.

## Commad Args
```
usage: llm_export.py [-h] --path PATH [--type TYPE] [--lora_path LORA_PATH] [--dst_path DST_PATH] [--test TEST] [--export EXPORT] [--skip_slim] [--quant_bit QUANT_BIT] [--quant_block QUANT_BLOCK]

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
```