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

#ifndef THIRD_PARTY_PY_AI_EDGE_TORCH_GENERATIVE_EXAMPLES_CPP_UTILS_H_
#define THIRD_PARTY_PY_AI_EDGE_TORCH_GENERATIVE_EXAMPLES_CPP_UTILS_H_

#include <cstddef>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/strings/string_view.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/signature_runner.h"
#include "tensorflow/lite/util.h"

namespace ai_edge_torch::examples {

// A minimal check macro.
#define MINIMAL_CHECK(x)                                     \
  if (!(x)) {                                                \
    fprintf(stderr, "Error at %s:%d\n", __FILE__, __LINE__); \
    exit(1);                                                 \
  }

// TF Lite requires all buffers (including external buffers used for KV cache
// here) be `tflite::kDefaultTensorAlignment` aligned. To ensure that, we use
// this custom allocator. Please use with caution as different platforms may
// have different alignment requirements.
template <typename T>
class AlignedAllocator {
 public:
  using value_type = T;

  T* allocate(std::size_t n) {
    void* ptr;
    std::size_t size = n * sizeof(T);
    std::size_t padding = tflite::kDefaultTensorAlignment -
                          (size % tflite::kDefaultTensorAlignment);
    size += padding;
    int ret = posix_memalign(&ptr, tflite::kDefaultTensorAlignment, size);
    if (ret != 0) {
      return nullptr;
    }
    return static_cast<T*>(ptr);
  };

  void deallocate(T* ptr, std::size_t n) { free(ptr); }
};

// An example implementation of LoRA adapters manager for TFLite interpreter.
// The class loads an adapter from a flatbuffers files and provides helper
// methods for finding the right signature and setting the appropriate input
// tensors. Please note the use of CustomAllocator to ensure zero-copy loading
// and potentially hot-swapping between multiple adapters with minimal cost.
class LoRA {
 public:
  static std::unique_ptr<LoRA> FromFile(absl::string_view path);

  tflite::SignatureRunner* GetPrefillRunner(tflite::Interpreter* interpreter,
                                            int matched_sequence_length) const;
  tflite::SignatureRunner* GetDecodeRunner(
      tflite::Interpreter* interpreter) const;

  int rank() const { return rank_; };

 private:
  explicit LoRA(int rank,
                absl::flat_hash_map<std::string,
                                    std::vector<float, AlignedAllocator<float>>>
                    tensors)
      : rank_(rank), tensors_(std::move(tensors)) {}

  tflite::SignatureRunner* GetRunnerHelper(
      tflite::Interpreter* interpreter, absl::string_view signature_name) const;

  // The rank of the LoRA adapter.
  const int rank_;
  // A Map of names to LoRA tensors.
  const absl::flat_hash_map<std::string,
                            std::vector<float, AlignedAllocator<float>>>
      tensors_;
};

}  // namespace ai_edge_torch::examples

#endif  // THIRD_PARTY_PY_AI_EDGE_TORCH_GENERATIVE_EXAMPLES_CPP_UTILS_H_
