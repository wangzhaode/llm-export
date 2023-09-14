# llm-export

llm-export是一个llm模型导出工具，能够将llm模型导出到onnx模型。

- 🚀 均完成`onnxruntime`正确性测试
- 🚀 优化原始代码，支持动态形状
- 🚀 优化原始代码，减少不必要常量


## 模型支持与下载
- ✅ [chatglm-6b](https://github.com/wangzhaode/llm-export/releases/tag/chatglm-6b-onnx)
- ✅ [chatglm2-6b](https://github.com/wangzhaode/llm-export/releases/tag/chatglm2-6b-onnx)
- ✅ [codegeex2-6b](https://github.com/wangzhaode/llm-export/releases/tag/codegeex2-6b-onnx)
- ✅ [Qwen-7B-Chat](https://github.com/wangzhaode/llm-export/releases/tag/qwen-7b-chat-onnx)
- ✅ [Baichuan2-7B-Chat](https://github.com/wangzhaode/llm-export/releases/tag/baichuan2-7b-chat-onnx)

## 用法
1. 将该项目clone到本地
```sh
git clnoe git@github.com:wangzhaode/LLMExporter.git
```
2. 将需要导出的LLM项目clone到本地，如：chatglm2-6b
```sh
git clone https://huggingface.co/THUDM/chatglm2-6b
# 如果huggingface下载慢可以使用modelscope
git clone https://modelscope.cn/ZhipuAI/chatglm2-6b.git
```
3. 执行LLMExporter导出模型
```sh
cd LLMExporter
python llm_export.py --path ../chatglm2-6b --export_path ./onnx --export
```

## 功能
- 支持将模型完整导出为一个onnx模型，使用`--export`
- 支持将模型分段导出为多个模型，使用`--export_split`
- 支持导出模型的词表到一个文本文件，每行代表一个token；其中token使用base64编码；使用`--export_verbose`
- 支持导出模型的Embedding层为一个onnx模型，使用`--export_embed`，同时支持bf16格式，使用`--embed_bf16`
- 支持分层导出模型的block，使用`--export_blocks`导出全部层；使用`--export_block $id`导出指定层
- 支持导出模型的lm_head层为一个onnx模型，使用`--export_lm`
- 支持对模型进行对话测试，使用`--test $query`会返回llm的回复内容
- 支持在导出onnx模型后使用onnxruntime对结果一致性进行校验，使用`--export_test`

## 参数
```
usage: llm_export.py [-h] --path PATH [--type {chatglm-6b,chatglm2-6b,Qwen-7B-Chat}] [--export_path EXPORT_PATH] [--export_verbose] [--export_test]
                     [--test TEST] [--export] [--export_split] [--export_vocab] [--export_embed] [--export_lm] [--export_block EXPORT_BLOCK]
                     [--export_blocks]

LLMExporter

optional arguments:
  -h, --help            show this help message and exit
  --path PATH           path(`str` or `os.PathLike`):
                        Can be either:
                        	- A string, the *model id* of a pretrained model like `THUDM/chatglm-6b`. [TODO]
                        	- A path to a *directory* clone from repo like `../chatglm-6b`.
  --type {chatglm-6b,chatglm2-6b,Qwen-7B-Chat}
                        type(`str`, *optional*):
                        	The pretrain llm model type.
  --export_path EXPORT_PATH
                        export onnx model path, defaut is `./onnx`.
  --export_verbose      Whether or not to export onnx with verbose.
  --export_test         Whether or not to export onnx with test using onnxruntime.
  --test TEST           test model inference with query `TEST`.
  --export              export model to an `onnx` model.
  --export_split        export model split to some `onnx` models:
                        	- embedding model.
                        	- block models.
                        	- lm_head model.
  --export_vocab        export llm vocab to a txt file.
  --export_embed        export llm embedding to an `onnx` model.
  --export_lm           export llm lm_head to an `onnx` model.
  --export_block EXPORT_BLOCK
                        export llm block [id] to an `onnx` model.
  --export_blocks       export llm all blocks to `onnx` models.
  --embed_bf16          using `bfloat16` replace `float32` in embedding.
```
