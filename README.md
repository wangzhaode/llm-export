# LLMExporter

LLMExporter是一个LLM模型导出工具，能够将LLM模型导出到onnx模型方便部署。

## 用法
1. 将该项目clone到本地
```sh
git clnoe git@github.com:wangzhaode/LLMExporter.git
```
2. 将需要导出的LLM项目clone到本地，如：chatglm2-6b
```sh
git clone https://huggingface.co/THUDM/chatglm2-6b
```
3. 执行LLMExporter导出模型
```sh
cd LLMExporter
python llm_export.py --path ../chatglm-6b --export_path ./onnx --export
```