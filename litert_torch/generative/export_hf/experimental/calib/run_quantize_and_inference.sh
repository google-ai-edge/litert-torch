#!/bin/bash
# Copyright 2026 The LiteRT Torch Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
# End-to-End Quantization and Inference Driver Script for LiteRT Torch in OSS.

set -e

# Default Paths (Modify as needed or override via env variables)
MODEL_PATH=${MODEL_PATH:-"/tmp/gemma3_models/model.tflite"}
EMBEDDER_MODEL_PATH=${EMBEDDER_MODEL_PATH:-"/tmp/gemma3_models/embedder.tflite"}
AUX_MODEL_PATH=${AUX_MODEL_PATH:-"/tmp/gemma3_models/aux.tflite"}
SPM_PATH=${SPM_PATH:-"/tmp/gemma3_models/tokenizer.model"}

DATASET_DIR=${DATASET_DIR:-"/tmp"}
DATASET_FORMAT=${DATASET_FORMAT:-"json"}
EVAL_TASK_NAMES=${EVAL_TASK_NAMES:-"smoke_check_prompts"}

CALIBRATION_DIR=${CALIBRATION_DIR:-"/tmp/calibration_oss"}
CALIBRATION_MERGED_DIR=${CALIBRATION_MERGED_DIR:-"/tmp/calibration_merged"}
QUANTIZED_DIR=${QUANTIZED_DIR:-"/tmp/quantized_oss"}

A16W8=${A16W8:-"true"}
ALIGN_KV_CACHE=${ALIGN_KV_CACHE:-"true"}
ALLOW_FLOAT_OPERATIONS=${ALLOW_FLOAT_OPERATIONS:-"true"}

MAX_DECODE_STEPS=${MAX_DECODE_STEPS:-"16"}
STOP_TOKEN=${STOP_TOKEN:-"106"}
PROMPT=${PROMPT:-"The capital of France is"}

# Interactive Configuration
configure_paths() {
  echo "Current Path Configurations:"
  echo "  MODEL_PATH           : ${MODEL_PATH}"
  echo "  EMBEDDER_MODEL_PATH  : ${EMBEDDER_MODEL_PATH}"
  echo "  AUX_MODEL_PATH       : ${AUX_MODEL_PATH}"
  echo "  SPM_PATH             : ${SPM_PATH}"
  echo "  QUANTIZED_DIR        : ${QUANTIZED_DIR}"
  echo "  DATASET_DIR          : ${DATASET_DIR}"
  echo ""
  read -p "Do you want to configure custom paths? [y/N]: " configure_custom
  if [[ "$configure_custom" =~ ^[yY]$ ]]; then
    read -p "Enter unquantized model path [${MODEL_PATH}]: " user_model
    MODEL_PATH=${user_model:-${MODEL_PATH}}

    read -p "Enter embedder model path [${EMBEDDER_MODEL_PATH}]: " user_embedder
    EMBEDDER_MODEL_PATH=${user_embedder:-${EMBEDDER_MODEL_PATH}}

    read -p "Enter auxiliary model path [${AUX_MODEL_PATH}]: " user_aux
    AUX_MODEL_PATH=${user_aux:-${AUX_MODEL_PATH}}

    read -p "Enter tokenizer/SPM model path [${SPM_PATH}]: " user_spm
    SPM_PATH=${user_spm:-${SPM_PATH}}

    read -p "Enter quantized output directory [${QUANTIZED_DIR}]: " user_quantized_dir
    QUANTIZED_DIR=${user_quantized_dir:-${QUANTIZED_DIR}}

    read -p "Enter calibration dataset directory [${DATASET_DIR}]: " user_dataset_dir
    DATASET_DIR=${user_dataset_dir:-${DATASET_DIR}}
  fi
}

# Helper Functions
run_inference_float() {
  echo "--- Running Float Model Inference..."
  python3 -m litert_torch.generative.export_hf.experimental.calib.sampling_executor_main \
    --model_path="${MODEL_PATH}" \
    --embedder_model_path="${EMBEDDER_MODEL_PATH}" \
    --auxiliary_model_path="${AUX_MODEL_PATH}" \
    --spm_path="${SPM_PATH}" \
    --prompt="${PROMPT}" \
    --enable_formatting=False \
    --max_decode_steps="${MAX_DECODE_STEPS}" \
    --stop_token="${STOP_TOKEN}" \
    --stream_output=True
}

run_calibration() {
  local output_save_dir=$1
  local task_names=$2
  echo "--- Running Calibration for task(s) [${task_names}] saving to ${output_save_dir}..."
  python3 -m litert_torch.generative.export_hf.experimental.calib.calibrate \
    --model_path="${MODEL_PATH}" \
    --embedder_model_path="${EMBEDDER_MODEL_PATH}" \
    --auxiliary_model_path="${AUX_MODEL_PATH}" \
    --spm_path="${SPM_PATH}" \
    --eval_task_names="${task_names}" \
    --dataset_dir="${DATASET_DIR}" \
    --dataset_format="${DATASET_FORMAT}" \
    --calibration_result_save_dir="${output_save_dir}" \
    --max_decode_steps=128
}

run_merge() {
  local input_dir=$1
  local output_dir=$2
  echo "--- Merging calibration results from ${input_dir} to ${output_dir}..."
  python3 -m litert_torch.generative.export_hf.experimental.calib.merge_calibration_results \
    --input_dir="${input_dir}" \
    --output_dir="${output_dir}"
}

run_quantization() {
  local calib_dir=$1
  echo "--- Running Quantization (a16w8=${A16W8}, align_kv=${ALIGN_KV_CACHE}, allow_float=${ALLOW_FLOAT_OPERATIONS}) using ${calib_dir}..."
  python3 -m litert_torch.generative.export_hf.experimental.calib.quantize \
    --model_path="${MODEL_PATH}" \
    --calibration_path="${calib_dir}/model.tflite.json" \
    --output_path="${QUANTIZED_DIR}/model.tflite" \
    --aux_model_path="${AUX_MODEL_PATH}" \
    --aux_calibration_path="${calib_dir}/aux.tflite.json" \
    --aux_output_path="${QUANTIZED_DIR}/aux.tflite" \
    --a16w8="${A16W8}" \
    --align_kv_cache="${ALIGN_KV_CACHE}" \
    --allow_float_operations="${ALLOW_FLOAT_OPERATIONS}"
}

run_inference_quantized() {
  echo "--- Running Quantized Model Inference..."
  python3 -m litert_torch.generative.export_hf.experimental.calib.sampling_executor_main \
    --model_path="${QUANTIZED_DIR}/model.tflite" \
    --embedder_model_path="${EMBEDDER_MODEL_PATH}" \
    --auxiliary_model_path="${QUANTIZED_DIR}/aux.tflite" \
    --spm_path="${SPM_PATH}" \
    --prompt="${PROMPT}" \
    --enable_formatting=False \
    --max_decode_steps="${MAX_DECODE_STEPS}" \
    --stop_token="${STOP_TOKEN}" \
    --stream_output=True
}

# Main Execution Switch
echo "=============================================================="
echo " Gemma 3 Quantization & Inference End-to-End Tool"
echo "=============================================================="
echo "Please select your Critical User Journey (CUJ):"
echo "  [0] Run Inference only (Float model)"
echo "  [1] Run End-to-End (Single-dataset Calibrate -> Quantize -> Inference)"
echo "  [2] Run Calibration only (Saves to task-specific sub-directory)"
echo "  [3] Run Merge & Quantize only (using completed task directories)"
echo "  [4] Run Inference only (Quantized model)"
echo "=============================================================="
read -p "Enter choice [0-4]: " choice

case $choice in
  0)
    configure_paths
    run_inference_float
    ;;
  1)
    configure_paths
    # Single-dataset path automatically wraps output in a sub-directory to enable merging
    task_save_dir="${CALIBRATION_DIR}/${EVAL_TASK_NAMES}"
    mkdir -p "${task_save_dir}"
    run_calibration "${task_save_dir}" "${EVAL_TASK_NAMES}"

    # Clean and merge the single task results
    rm -rf "${CALIBRATION_MERGED_DIR}"
    run_merge "${CALIBRATION_DIR}" "${CALIBRATION_MERGED_DIR}"

    run_quantization "${CALIBRATION_MERGED_DIR}"
    run_inference_quantized
    ;;
  2)
    configure_paths
    read -p "Enter custom calibration task name [${EVAL_TASK_NAMES}]: " user_task
    active_task=${user_task:-${EVAL_TASK_NAMES}}

    task_save_dir="${CALIBRATION_DIR}/${active_task}"
    mkdir -p "${task_save_dir}"
    run_calibration "${task_save_dir}" "${active_task}"
    echo "--- Calibration completed successfully! Results saved to: ${task_save_dir}"
    echo "--- You can run this script in other terminals to calibrate other tasks in parallel."
    ;;
  3)
    configure_paths
    read -p "Enter parent calibration directory [${CALIBRATION_DIR}]: " user_calib_dir
    active_parent=${user_calib_dir:-${CALIBRATION_DIR}}

    # Automatically runs the merge step across all completed tasks first
    rm -rf "${CALIBRATION_MERGED_DIR}"
    run_merge "${active_parent}" "${CALIBRATION_MERGED_DIR}"

    run_quantization "${CALIBRATION_MERGED_DIR}"
    run_inference_quantized
    ;;
  4)
    configure_paths
    run_inference_quantized
    ;;
  *)
    echo "Invalid choice!"
    exit 1
    ;;
esac
