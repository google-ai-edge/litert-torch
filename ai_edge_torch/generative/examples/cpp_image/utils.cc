/* Copyright 2025 The AI Edge Torch Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "ai_edge_torch/generative/examples/cpp_image/utils.h"

#include <cstddef>
#include <cstring>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/memory/memory.h"
#include "absl/strings/match.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/model_builder.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/signature_runner.h"

namespace ai_edge_torch::examples {

std::unique_ptr<LoRA> LoRA::FromFile(absl::string_view path) {
  std::unique_ptr<tflite::FlatBufferModel> model =
      tflite::FlatBufferModel::VerifyAndBuildFromFile(path.data());
  if (model == nullptr) {
    return nullptr;
  }

  int rank = -1;
  absl::flat_hash_map<std::string, std::vector<float, AlignedAllocator<float>>>
      tensors;
  for (const auto& tensor :
       *model->GetModel()->subgraphs()->Get(0)->tensors()) {
    size_t size = 1;
    for (const int& dim : *tensor->shape()) {
      size *= dim;
    }
    std::vector<float, AlignedAllocator<float>> buffer(size);
    const auto* data =
        model->GetModel()->buffers()->Get(tensor->buffer())->data();
    memcpy(buffer.data(), data->data(), data->size());
    tensors.emplace(*tensor->name(), std::move(buffer));

    if (tensor->name()->str() == "lora_atten_q_a_prime_weight_0") {
      rank = tensor->shape()->Get(1);
    }
  }
  if (rank == -1) {
    return nullptr;
  }

  return absl::WrapUnique(new LoRA(rank, std::move(tensors)));
}

tflite::SignatureRunner* LoRA::GetPrefillRunner(
    tflite::Interpreter* interpreter, int matched_sequence_length) const {
  std::string signature_name =
      absl::StrFormat("prefill_%d_lora_r%d", matched_sequence_length, rank_);
  return GetRunnerHelper(interpreter, signature_name);
}

tflite::SignatureRunner* LoRA::GetDecodeRunner(
    tflite::Interpreter* interpreter) const {
  std::string signature_name = absl::StrFormat("decode_lora_r%d", rank_);
  return GetRunnerHelper(interpreter, signature_name);
};

tflite::SignatureRunner* LoRA::GetRunnerHelper(
    tflite::Interpreter* interpreter, absl::string_view signature_name) const {
  tflite::SignatureRunner* runner =
      interpreter->GetSignatureRunner(signature_name.data());
  if (runner == nullptr) {
    return nullptr;
  }

  absl::flat_hash_set<std::string> lora_input_tensors;
  lora_input_tensors.reserve(runner->input_size());
  for (const char* input_name : runner->input_names()) {
    if (absl::StrContains(input_name, "lora")) {
      lora_input_tensors.insert(input_name);
    }
  }

  if (lora_input_tensors.size() < tensors_.size()) {
    return nullptr;
  }

  for (const auto& [name, buffer] : tensors_) {
    TfLiteTensor* tensor = runner->input_tensor(name.c_str());
    if (tensor == nullptr) {
      return nullptr;
    }
    lora_input_tensors.erase(name);
    TfLiteCustomAllocation allocation = {
        .data = static_cast<void*>(const_cast<float*>(buffer.data())),
        .bytes = buffer.size() * sizeof(float)};
    if (runner->SetCustomAllocationForInputTensor(name.c_str(), allocation) !=
        kTfLiteOk) {
      return nullptr;
    }
  };
  if (runner->AllocateTensors() != kTfLiteOk) {
    return nullptr;
  }

  for (const auto& name : lora_input_tensors) {
    TfLiteTensor* tensor = runner->input_tensor(name.c_str());
    if (tensor == nullptr) {
      return nullptr;
    }
    memset(tensor->data.data, 0, tensor->bytes);
  }

  return runner;
}

}  // namespace ai_edge_torch::examples
