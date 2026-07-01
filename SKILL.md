---
name: converting-pytorch-to-litert
description: >-
  Converts PyTorch models (e.g. ResNet, timm, HuggingFace transformers) directly to
  LiteRT (.tflite) flatbuffer format. Use when converting PyTorch models to TFLite,
  setting up export environments, or troubleshooting torch-to-litert conversion bugs.
  Don't use for ONNX exports or converting existing TensorFlow models.
---

# PyTorch to LiteRT Conversion Guide

**ATTENTION ALL AGENTS:** If you are tasked with converting a PyTorch model (e.g., ResNet or similar architectures) to a `.tflite` flatbuffer format in this repository, **you must strictly follow these rules based on recent project findings:**

## 1. Direct Conversion Only (No ONNX)
Bypass ONNX entirely. Do not attempt to export the model to ONNX or use `onnx2tf`. Convert directly from PyTorch to LiteRT.

## 2. Use `litert-torch`
Use the `litert-torch` library for the direct PyTorch to TFLite conversion. Note that `ai-edge-torch` is deprecated and has been officially renamed to `litert-torch`.

## 3. Strict Python Environment Rules
- **Do NOT use Python 3.14**. It currently has compatibility issues with typing in `torchao` (`AttributeError: 'typing.Union' object has no attribute '__module__'`).
- **Use Python 3.11** (e.g., `python3.11`) as it is highly recommended for stability and avoids the aforementioned type hint bugs.

## 4. Dependencies
Before running any conversion script, ensure the following are installed in your `python3.11` environment:
```bash
python3.11 -m pip install torch transformers timm litert-torch ai-edge-litert litert-cli-nightly