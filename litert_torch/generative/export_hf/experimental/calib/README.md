# LiteRT Torch LLM Calibration & Quantization Tool Suite

This folder contains the official tools to calibrate, merge, and statically quantize LiteRT LLM models (such as Gemma 3) for high-performance edge and NPU deployments.

---

## Quick Start: E2E Interactive Driver Script

The easiest way to run the complete pipeline is using the interactive driver script:

```bash
# 1. Navigate to the target directory
cd litert_torch/generative/export_hf/experimental/calib/

# 2. Make it executable (if needed)
chmod +x run_quantize_and_inference.sh

# 3. Launch the tool!
./run_quantize_and_inference.sh
```

The script will present an interactive menu with **5 Critical User Journeys (CUJs)**:

*   **`[0] Float Model Inference`**: Instantly runs text generation on the unquantized Float32 model.
*   **`[1] End-to-End (Single Dataset)`**: Executes single-task Calibration, automatically merges results, runs advanced quantization (with full KV cache alignment and float protections active), and immediately launches the quantized inference test.
*   **`[2] Calibration Only`**: Prompts for a custom task name, runs calibration, and exits cleanly. This allows you to open multiple parallel terminal tabs to run calibrations for different tasks concurrently.
*   **`[3] Merge & Quantize Only`**: Discovers all completed task sub-directories under your calibration directory, merges them, aligns KV cache scaling parameters, statically quantizes both the main and auxiliary models, and runs inference. (Use this when tuning quantization parameters without re-running calibrations).
*   **`[4] Quantized Model Inference`**: Directly executes text generation on your newly quantized model suite.

*Note: At startup, you will be prompted: `Do you want to configure custom paths? [y/N]:`. Type `y` to configure custom LiteRT model paths, tokenizers, datasets, or output directories.*

---

## Running Python Scripts Separately

If you want to trigger individual stages of the pipeline manually, you can invoke each Python script separately in your virtual environment.

### 1. Calibration (`calibrate.py`)
Calibrates the model against a flat JSON list of conversational prompts to calculate min/max activation ranges. You can optionally enable profiler-based calibration for more accurate activation tracking:

```bash
python3 -m litert_torch.generative.export_hf.experimental.calib.calibrate \
  --model_path=/tmp/gemma3_models/model.tflite \
  --embedder_model_path=/tmp/gemma3_models/embedder.tflite \
  --auxiliary_model_path=/tmp/gemma3_models/aux.tflite \
  --spm_path=/tmp/gemma3_models/tokenizer.model \
  --eval_task_names="my_custom_task" \
  --dataset_dir=/tmp \
  --dataset_format=json \
  --calibration_result_save_dir=/tmp/calibration_oss/my_custom_task \
  --max_decode_steps=128 \
  --use_profiler_based_calibration=True \
  --enable_min_max_calibration_update=True
```
*Note: `--use_profiler_based_calibration=True` runs calibration in profiler-based mode, which collects accurate min/max ranges from the interpreter's internal profiler. This is highly recommended and requires `--enable_min_max_calibration_update=True` to update the ranges correctly.*
*Note: Keep task JSON outputs grouped inside separate sub-folders (e.g. `/tmp/calibration_oss/my_custom_task/`) to enable the merger to discover them.*

### 2. Merging Calibration Results (`merge_calibration_results.py`)
Combines min/max ranges across multiple task sub-folders into a single aligned calibration JSON:

```bash
python3 -m litert_torch.generative.export_hf.experimental.calib.merge_calibration_results \
  --input_dir=/tmp/calibration_oss \
  --output_dir=/tmp/calibration_merged
```

### 3. Advanced Quantization (`quantize.py`)
Statically quantizes both the main and auxiliary models. This script natively supports high-precision configurations:
*   `--a16w8`: Use `True` for 16-bit activation quantization, `False` for standard 8-bit activations.
*   `--align_kv_cache`: Align Key-Value cache parameters across main and auxiliary subgraphs.
*   `--allow_float_operations`: Protects RMS Norm, residual additions, and skip scales in Float32 to maintain generation quality.
*   `--skip_mlir_passes`: Use `True` to skip post-quantization MLIR graph surgery passes (required for some models like Kanana).
*   `--calibration_range_scale`: Scale factor (e.g. `1.15`) applied to the calibration min/max ranges before quantization to prevent clipping.

```bash
python3 -m litert_torch.generative.export_hf.experimental.calib.quantize \
  --model_path=/tmp/gemma3_models/model.tflite \
  --calibration_path=/tmp/calibration_merged/model.tflite.json \
  --output_path=/tmp/quantized_oss/model.tflite \
  --aux_model_path=/tmp/gemma3_models/aux.tflite \
  --aux_calibration_path=/tmp/calibration_merged/aux.tflite.json \
  --aux_output_path=/tmp/quantized_oss/aux.tflite \
  --a16w8=True \
  --align_kv_cache=True \
  --allow_float_operations=True \
  --skip_mlir_passes=True \
  --calibration_range_scale=1.15
```
*Note: `embedder.tflite` must NOT be quantized. Copy the unquantized Float32 embedder directly to your output folder.*

### 4. Text Generation / Inference (`sampling_executor_main.py`)
Executes text generation testing on the CPU:

```bash
python3 -m litert_torch.generative.export_hf.experimental.calib.sampling_executor_main \
  --model_path=/tmp/quantized_oss/model.tflite \
  --embedder_model_path=/tmp/gemma3_models/embedder.tflite \
  --auxiliary_model_path=/tmp/quantized_oss/aux.tflite \
  --spm_path=/tmp/gemma3_models/tokenizer.model \
  --prompt="The capital of France is" \
  --enable_formatting=False \
  --max_decode_steps=16 \
  --stop_token=106 \
  --stream_output=True
```
*Note: Pass `--stop_token=106` for Gemma 3 models to cleanly halt generation the moment the model finishes its conversational turn.*
