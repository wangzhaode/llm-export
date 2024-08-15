# llm-export

[English](./README_en.md)

llm-export是一个llm模型导出工具，能够将llm模型导出为onnx和mnn模型。

- 🚀 优化原始代码，支持动态形状
- 🚀 优化原始代码，减少常量部分
- 🚀 使用[OnnxSlim](https://github.com/inisis/OnnxSlim)优化onnx模型，性能提升约5%; by [@inisis](https://github.com/inisis)
- 🚀 支持将lora权重导出为onnx和mnn
- 🚀 Onnx推理代码[OnnxLLM](https://github.com/inisis/OnnxLLM)

## 用法
1. 将该项目clone到本地
```sh
git clone git@github.com:wangzhaode/llm-export.git
```
2. 将需要导出的LLM项目clone到本地，如：chatglm2-6b
```sh
git clone https://huggingface.co/THUDM/chatglm2-6b
# 如果huggingface下载慢可以使用modelscope
git clone https://modelscope.cn/ZhipuAI/chatglm2-6b.git
```
3. 导出模型
```sh
cd mnn-llm
# 将chatglm2-6b导出为onnx模型
python llm_export.py --path ../chatglm2-6b --export onnx
# 将chatglm2-6b导出为mnn模型, 量化参数为4bit, blokc-wise = 128
python llm_export.py --path ../chatglm2-6b --export mnn --quant_bit 4 --quant_block 128
```

## 功能
- 支持将模型为onnx或mnn模型，使用`--export onnx`或`--export mnn`
- 支持对模型进行对话测试，使用`--test $query`会返回llm的回复内容
- 默认会使用onnx-slim对onnx模型进行优化，跳过该步骤使用`--skip_slim`
- 支持合并lora权重后导出，指定lora权重的目录使用`--lora_path`

## 参数
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
