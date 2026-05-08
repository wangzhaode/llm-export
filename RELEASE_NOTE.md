# Release Notes v0.0.5

## 架构重构

### 1. 模块拆分
- 将 `llmexport.py` 的核心逻辑拆分到独立的 `utils/` 子模块
- 新增 `utils/config.py`：统一 `LlmConfig` 模型配置定义
- 新增 `utils/model.py`：独立的 `LlmModel` 包装类
- 新增 `utils/tokenizer.py`：统一的 `LlmTokenizer` 封装
- 主入口代码精简近一半，结构更清晰，便于扩展与维护

## 新增功能

### 1. 新增模型支持
- **LFM2 系列**：新增对 Liquid LFM2 的全面支持，包括文本（`lfm2`）、MoE（`lfm2_moe`）、视觉多模态（`lfm2_vl`）和音频多模态（`lfm2_audio`）
- **Gemma 4**：新增对 Gemma 4 的支持
- **Qwen3.5**：新增对 Qwen3.5 的支持
- **Qwen2-Audio**：新增对 Qwen2-Audio 音频多模态模型的支持
- **FunAudio**：新增对 FunAudioChat 的支持
- **GLM-OCR**：新增对 GLM 系列 OCR 多模态模型的支持
- 合并原 `phi3` 到统一的 `phi` 映射

### 2. Safetensors 直转 MNN
- 新增 `safetensors2mnn.py`，支持从 safetensors 权重直接导出为 MNN 格式
- 无需先转 ONNX，缩短导出链路

### 3. OmniQuantizer 量化器
- 新增 `utils/omni_quantizer.py`，提供基于优化的端到端量化方案
- 支持权重与激活联合校准，可配置 epochs / lr / weight decay
- 支持为 NPU 目标生成特定量化格式

### 4. 多模态能力增强
- `utils/vision.py` 大幅扩充，适配更多视觉编码器结构
- `utils/audio.py` 完善音频模型处理流程
- `utils/transformers.py` 扩展 Decoder / Embedding / Rotary 以支持更多架构变体

## 修复问题

### 1. ONNX Slice 索引修复
- 修复 ONNX 导出中 slice 索引越界问题

## 优化改进

### 1. 量化算法增强
- `smooth_quantizer.py` 重写，量化精度与稳定性提升
- `hqq_quantizer.py` 优化量化流程
- 自定义算子 `custom_op.py` 扩展，支持更多量化场景

### 2. 依赖管理
- 新增 `requirements.txt`，明确最小依赖集合

## 使用示例

```bash
# 从 safetensors 直接导出为 MNN
python -m llmexport.safetensors2mnn --path /path/to/model

# 使用 Omni 量化导出
llmexport --path Qwen2.5-1.5B-Instruct --export mnn --quant_bit 4 --quant_block 128
```

## 兼容性说明

- 向后兼容 v0.0.4 版本命令行接口
- 内部模块路径重构，若有直接 import `llmexport` 子模块的代码需调整
