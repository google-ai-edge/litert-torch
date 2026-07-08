---
name: litert-quantization-calib
description: >-
  Assists the user to calibrate, merge, and statically quantize litert LLM models
  (such as Gemma 3) in standard open-source (OSS) environments.
  Use when the user wants to run LLM calibration, merge task JSON results,
  align KV cache parameters across models, protect sensitive layers in Float32,
  or run quantized inference testing.
  Don't use for JAX/PyTorch custom quantization configurations or non-litert models.
---

# LiteRT LLM Calibration & Quantization Skill

A skill to guide the calibration, alignment, precision protection, and static
range quantization of LiteRT LLMs (such as Gemma 3 1B) for high-performance NPU
deployments.

--------------------------------------------------------------------------------

## Triggers and Anti-Triggers

*   **Use when**:

    *   The user wants to calibrate a LiteRT model against conversational prompt
        datasets.
    *   The user needs to merge calibration JSON files from multiple different
        tasks.
    *   The user wants to statically quantize a model suite (e.g. A8W8 or A16W8
        configurations).
    *   The user needs to protect RMS Norm or residual paths from quantization
        noise in Float32.
    *   The user needs to align KV Cache quantization scaling parameters across
        main and auxiliary models.
    *   The user wants to run CPU/NPU inference testing on unquantized or
        quantized LiteRT models.

*   **Don't use for**:

    *   Quantizing non-LiteRT models (e.g. PyTorch native, JAX native, or ONNX
        models).
    *   General quantizations that do not involve LLM prefill/decode subgraphs
        or KV caches.

--------------------------------------------------------------------------------

## Principles & Mathematical Gotchas

Every LLM NPU quantization task has highly sensitive numerical paths. You
**must** strictly adhere to these verified gotchas to prevent quality
degradation or loop crashes:

### 1. Mandatory BOS (Beginning of Sequence) Token

*   **Gotcha**: Tokenizing prompts without the BOS token (`2`) will completely
    corrupt the model's attention states and positional encodings at step 0.
    This causes the model to generate garbage loop outputs (e.g. `does France
    does France` infinitely) during both calibration and inference.
*   **Rule**: You **must** ensure that `prepend_bos=True` is active during all
    tokenization operations. If using the open-source SentencePiece or
    Transformers tokenizers, verify that the BOS token ID (`2`) is explicitly
    prepended to the start of the token IDs list.

### 2. Precision Protection in Float32 (`allow_float_operations=True`)

*   **Gotcha**: Statically quantizing RMS Norm operations, residual layer
    additions, or skip scale multipliers down to Int8/Int16 introduces severe
    quantization noise that ruins text generation quality.
*   **Rule**: Keep these three mathematical paths protected in **Float32**:
    1.  **RMS Norm / Layer Norm operations** (typically the StableHLO composite
        or norm nodes): Skip from quantization to prevent numerical accuracy
        loss.
    2.  **Residual connection additions** (typically the element-wise add nodes
        immediately following attention or MLP projection blocks): Keep in
        Float32.
    3.  **Residual connection scaling multipliers** (typically the element-wise
        multiplication nodes matching the skip connection scaling factors): Keep
        in Float32.

### 3. Key-Value (KV) Cache Parameter Alignment (`align_kv_cache=True`)

*   **Gotcha**: Key and Value cache tensors across the main model
    (prefill/decode subgraphs) and auxiliary model must use identical scaling
    factors (scale and zero-point) so the hardware NPU can reuse slices
    dynamically without expensive runtime rescalings.
*   **Rule**: Always execute `align_kv_cache_params()` to search and align all K
    and V cache layers across the model suite before calling the quantizer.

### 4. Keep `embedder.tflite` in Float32 (Never Quantize)

*   **Gotcha**: The embedder model contains a massive lookup table. Quantizing
    its weights leads to massive accuracy degradation. Since the embedder only
    executes once at step 0 (prefill), it has zero speed impact.
*   **Rule**: Do **not** quantize `embedder.tflite`. Copy the original Float32
    version directly into the final quantized deployment folder.

### 5. Profiler-Based Calibration (`use_profiler_based_calibration=True`)

*   **Gotcha**: Standard calibration runs inference and collects activation statistics from outputs. However, some internal activation bounds might be missed or underestimated.
*   **Rule**: Use `--use_profiler_based_calibration=True` to run the calibration interpreter in profiler mode. This extracts precise tensor bounds directly from the interpreter's internal profiling metrics. This mode *requires* `--enable_min_max_calibration_update=True` to correctly update the min/max calibration parameters.

### 6. Skipping MLIR Passes and Scaling Ranges

*   **Gotcha**: Standard post-quantization MLIR graph surgery passes can sometimes fail or crash on custom architectures (like Kanana's swapped KV cache). Additionally, tight min/max calibration bounds can cause clipping of activation outliers, degrading model quality.
*   **Rule**:
    *   Use `--skip_mlir_passes=True` to bypass failing MLIR optimizations for sensitive custom model architectures.
    *   Use `--calibration_range_scale=1.15` (or a similar factor) to slightly scale up the calibration bounds, providing headroom for outliers and reducing clipping distortion.

--------------------------------------------------------------------------------

## Core Command Patterns

### 1. Open-Source Calibration Run

Generates the calibration range JSONs under a task-specific sub-directory:

```bash
python3 -m litert_torch.generative.export_hf.experimental.calib.calibrate \
  --model_path={model_path} \
  --embedder_model_path={embedder_path} \
  --auxiliary_model_path={aux_path} \
  --spm_path={tokenizer_spm_path} \
  --eval_task_names="{task_name}" \
  --dataset_dir={dataset_dir} \
  --dataset_format=json \
  --calibration_result_save_dir={output_save_dir}/{task_name} \
  --max_decode_steps=128 \
  --use_profiler_based_calibration=True \
  --enable_min_max_calibration_update=True
```

### 2. Merging Calibration Results

Discovers and merges QSV files across task sub-directories into a single aligned calibration file:

```bash
python3 -m litert_torch.generative.export_hf.experimental.calib.merge_calibration_results \
  --input_dir={calibration_dir} \
  --output_dir={merged_output_dir}
```
*Note: The merger expects a directory structure where task files are grouped inside sub-folders (e.g. `input_dir/task_name/model.tflite.json`). Flat folders will result in no tasks discovered.*

### 3. Advanced Multi-Model Quantization (A8W8 or A16W8)

Statically quantizes both the main model and the auxiliary model with KV Cache Alignment and Float Protection active:

```bash
python3 -m litert_torch.generative.export_hf.experimental.calib.quantize \
  --model_path={model_path} \
  --calibration_path={merged_dir}/model.tflite.json \
  --output_path={quantized_dir}/model.tflite \
  --aux_model_path={aux_path} \
  --aux_calibration_path={merged_dir}/aux.tflite.json \
  --aux_output_path={quantized_dir}/aux.tflite \
  --a16w8={true_or_false} \
  --align_kv_cache=True \
  --allow_float_operations=True \
  --skip_mlir_passes=True \
  --calibration_range_scale=1.15
```

### 4. Quantized Inference Testing

Executes text generation using the quantized model suite:

```bash
python3 -m litert_torch.generative.export_hf.experimental.calib.sampling_executor_main \
  --model_path={quantized_dir}/model.tflite \
  --embedder_model_path={embedder_path} \
  --auxiliary_model_path={quantized_dir}/aux.tflite \
  --spm_path={tokenizer_spm_path} \
  --prompt="{prompt_text}" \
  --enable_formatting=False \
  --max_decode_steps=16 \
  --stop_token={stop_token_id} \
  --stream_output=True
```
*Note: For Gemma 3, set `--stop_token=106` to ensure the executor cleanly halts immediately when the model generates its end-of-turn sequence.*

--------------------------------------------------------------------------------

## The E2E Interactive Driver tool

For the easiest, most robust end-to-end execution, you can use the interactive
bash script inside the repository: `bash cd
litert_torch/generative/export_hf/experimental/calib/
./run_quantize_and_inference.sh` This script provides interactive path
configuration, parallel calibration task management, automatic task grouping,
merging, and A16W8/A8W8 recipe quantization!
