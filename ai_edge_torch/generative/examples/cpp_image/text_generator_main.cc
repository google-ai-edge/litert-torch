/* Copyright 2024 The AI Edge Torch Authors. All Rights Reserved.

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

#include <algorithm>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <ios>
#include <iterator>
#include <limits>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/strings/match.h"
#include "ai_edge_torch/generative/examples/cpp_image/utils.h"

// STB Image library - header-only libraries for image loading and resizing
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include "stb_image_resize2.h"  // Using the newer resize API
#include "src/sentencepiece_processor.h"
#include "tensorflow/lite/delegates/xnnpack/xnnpack_delegate.h"
#include "tensorflow/lite/experimental/genai/genai_ops.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/interpreter_builder.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model_builder.h"
#include "tensorflow/lite/signature_runner.h"

// This is a simplified example of using TFLite to generate text.
// Please note that this is only a starting point and the user is expected
// to create their own pipeline potentially using different tokenizers and
// better samplers.
//
// Example usage:
//  Text only:
//  generate_main --tflite_model="PATH/model.tflite" \
//    --sentencepiece_model="PATH/sp.model" \
//    --prompt="Write an email:" \
//    --max_generated_tokens=64 \
//    --start_token="<bos>" \
//    --stop_token="<eos>"  \
//    --num_threads=4
//
//  With image input:
//  generate_main --tflite_model="PATH/model.tflite" \
//    --sentencepiece_model="PATH/sp.model" \
//    --prompt="<image>Describe this image:" \
//    --max_generated_tokens=64 \
//    --start_token="<bos>" \
//    --stop_token="<eos>"  \
//    --num_threads=4 \
//    --use_single_image=true \
//    --image_path="PATH/image.jpg"

ABSL_FLAG(std::string, tflite_model, "",
          "Two-signature tflite model prepared for text generation using ODML "
          "tools.");
ABSL_FLAG(std::string, sentencepiece_model, "", "Path to sentencepiece model.");
ABSL_FLAG(std::string, prompt, "Write an email:", "Input prompt to the model.");
ABSL_FLAG(int, max_decode_steps, -1,
          "The number of tokens to generate. Defaults to the KV cache size "
          "defined during conversion.");
ABSL_FLAG(int, max_generated_tokens, -1,
          "Maximum number of tokens to generate. Defaults to the KV cache size "
          "defined during conversion. Takes precedence over max_decode_steps if set.");
ABSL_FLAG(std::string, start_token, "",
          "Start token is appended to the beginning of input prompt to "
          "signify start of sentence.");
ABSL_FLAG(std::string, stop_token, "",
          "Stop token used to deterine end of decoding loop. If not provided "
          "will decode until max_kv_cache_size or max_decode_steps.");
ABSL_FLAG(int, num_threads, 4, "Number of threads to use. Defaults to 4.");
ABSL_FLAG(std::string, weight_cache_path, "",
          "XNNPACK weight caching path, e.g. /tmp/model.xnnpack_cache.");
ABSL_FLAG(std::string, lora_path, "", "Optional path to LoRA artifact.");
ABSL_FLAG(bool, use_pixel_prefill, true, 
          "Whether to prefer prefill signatures with pixel support. "
          "Set to false to force text-only prefill signatures.");
ABSL_FLAG(bool, use_single_image, true,
          "Whether to use single image mode (true) or split image mode (false). "
          "Single image mode uses simpler token pattern, split image mode uses 3x4 grid.");
ABSL_FLAG(std::string, image_path, "",
          "Path to input image file (jpg, png, etc.). If empty, dummy pixel values will be used.");

namespace {

using ai_edge_torch::examples::AlignedAllocator;
using ai_edge_torch::examples::LoRA;

// Image processing constants
constexpr int TARGET_IMAGE_SIZE = 512;
constexpr int NUM_CHANNELS = 3;
constexpr float IMAGE_MEAN = 0.5f;
constexpr float IMAGE_STD = 0.5f;

// Load and process image for model input
bool LoadAndProcessImage(const std::string& image_path, TfLiteTensor* pixel_values_tensor) {
  if (image_path.empty() || pixel_values_tensor == nullptr) {
    return false;
  }

  // Load image using stb_image
  int width, height, channels;
  unsigned char* image_data = stbi_load(image_path.c_str(), &width, &height, &channels, NUM_CHANNELS);
  if (!image_data) {
    printf("Failed to load image: %s\n", image_path.c_str());
    return false;
  }

  printf("Loaded image: %dx%d with %d channels\n", width, height, channels);

  // Resize image to 512x512
  std::vector<unsigned char> resized_image(TARGET_IMAGE_SIZE * TARGET_IMAGE_SIZE * NUM_CHANNELS);
  if (!stbir_resize_uint8_linear(image_data, width, height, width * NUM_CHANNELS,
                                 resized_image.data(), TARGET_IMAGE_SIZE, TARGET_IMAGE_SIZE, 
                                 TARGET_IMAGE_SIZE * NUM_CHANNELS, 
                                 STBIR_RGB)) {
    printf("Failed to resize image\n");
    stbi_image_free(image_data);
    return false;
  }

  printf("Resized image to %dx%d\n", TARGET_IMAGE_SIZE, TARGET_IMAGE_SIZE);

  // Get tensor dimensions
  int batch = pixel_values_tensor->dims->data[0];
  int n_images = pixel_values_tensor->dims->data[1];
  int channels_dim = pixel_values_tensor->dims->data[2];
  int height_dim = pixel_values_tensor->dims->data[3];
  int width_dim = pixel_values_tensor->dims->data[4];

  printf("Tensor shape: [%d, %d, %d, %d, %d]\n", batch, n_images, channels_dim, height_dim, width_dim);

  // Verify tensor dimensions match expected format [1, 1, 3, 512, 512]
  if (batch != 1 || n_images != 1 || channels_dim != NUM_CHANNELS ||
      height_dim != TARGET_IMAGE_SIZE || width_dim != TARGET_IMAGE_SIZE) {
    printf("Error: Tensor dimensions don't match expected format [1, 1, 3, 512, 512]\n");
    stbi_image_free(image_data);
    return false;
  }

  // Convert to float, normalize, and fill tensor
  // Formula: (pixel_value / 255.0 - mean) / std
  float* tensor_data = pixel_values_tensor->data.f;
  
  // Debug: Print first few raw pixel values before normalization
  printf("Raw pixel values (first 10): ");
  for (int i = 0; i < 10 && i < TARGET_IMAGE_SIZE * TARGET_IMAGE_SIZE * NUM_CHANNELS; ++i) {
    printf("%d ", resized_image[i]);
  }
  printf("\n");
  
  for (int c = 0; c < NUM_CHANNELS; ++c) {
    for (int h = 0; h < TARGET_IMAGE_SIZE; ++h) {
      for (int w = 0; w < TARGET_IMAGE_SIZE; ++w) {
        // Input image is HWC format, tensor is [batch, n_images, channels, height, width]
        int input_idx = (h * TARGET_IMAGE_SIZE + w) * NUM_CHANNELS + c;
        int tensor_idx = c * (TARGET_IMAGE_SIZE * TARGET_IMAGE_SIZE) + h * TARGET_IMAGE_SIZE + w;
        
        // Convert to float [0, 1], then normalize with mean=0.5, std=0.5
        // Using double precision for intermediate calculation to improve accuracy
        double pixel_value = static_cast<double>(resized_image[input_idx]) / 255.0;
        tensor_data[tensor_idx] = static_cast<float>((pixel_value - IMAGE_MEAN) / IMAGE_STD);
      }
    }
  }
  
  // Debug: Print first few normalized values
  printf("Normalized values (first 10): ");
  for (int i = 0; i < 10; ++i) {
    printf("%.6f ", tensor_data[i]);
  }
  printf("\n");

  printf("Successfully processed image and filled tensor with normalized values\n");
  
  // Print tensor values in PyTorch format
  printf("\ntensor([[[[");
  for (int c = 0; c < NUM_CHANNELS; ++c) {
    if (c > 0) printf("\n\n          ");
    printf("[");
    
    // Print first 3 rows
    for (int h = 0; h < 3 && h < TARGET_IMAGE_SIZE; ++h) {
      if (h > 0) printf("\n           ");
      printf("[");
      
      // Print first 3 values
      for (int w = 0; w < 3 && w < TARGET_IMAGE_SIZE; ++w) {
        int tensor_idx = c * (TARGET_IMAGE_SIZE * TARGET_IMAGE_SIZE) + h * TARGET_IMAGE_SIZE + w;
        printf("%.4f", tensor_data[tensor_idx]);
        if (w < 2) printf(", ");
      }
      
      printf(",  ..., ");
      
      // Print last 3 values
      for (int w = TARGET_IMAGE_SIZE - 3; w < TARGET_IMAGE_SIZE; ++w) {
        int tensor_idx = c * (TARGET_IMAGE_SIZE * TARGET_IMAGE_SIZE) + h * TARGET_IMAGE_SIZE + w;
        printf("%.4f", tensor_data[tensor_idx]);
        if (w < TARGET_IMAGE_SIZE - 1) printf(", ");
      }
      
      printf("]");
      if (h < 2) printf(",");
    }
    
    printf(",\n           ...,\n");
    
    // Print last 3 rows
    for (int h = TARGET_IMAGE_SIZE - 3; h < TARGET_IMAGE_SIZE; ++h) {
      printf("           [");
      
      // Print first 3 values
      for (int w = 0; w < 3 && w < TARGET_IMAGE_SIZE; ++w) {
        int tensor_idx = c * (TARGET_IMAGE_SIZE * TARGET_IMAGE_SIZE) + h * TARGET_IMAGE_SIZE + w;
        printf("%.4f", tensor_data[tensor_idx]);
        if (w < 2) printf(", ");
      }
      
      printf(",  ..., ");
      
      // Print last 3 values
      for (int w = TARGET_IMAGE_SIZE - 3; w < TARGET_IMAGE_SIZE; ++w) {
        int tensor_idx = c * (TARGET_IMAGE_SIZE * TARGET_IMAGE_SIZE) + h * TARGET_IMAGE_SIZE + w;
        printf("%.4f", tensor_data[tensor_idx]);
        if (w < TARGET_IMAGE_SIZE - 1) printf(", ");
      }
      
      printf("]");
      if (h < TARGET_IMAGE_SIZE - 1) printf(",");
      printf("\n");
    }
    
    printf("          ]");
    if (c < NUM_CHANNELS - 1) printf(",");
  }
  printf("]]))\n");
  
  stbi_image_free(image_data);
  return true;
}

// Function to generate expanded image tokens for when the image is split into patches
std::string PromptSplitImage(
    int image_seq_len, int image_rows, int image_cols,
    const std::string& fake_token_around_image,
    const std::string& image_token,
    const std::string& global_image_token) {
  std::string text_split_images = "";
  
  // Generate grid pattern with row/col tokens
  for (int n_h = 0; n_h < image_rows; ++n_h) {
    for (int n_w = 0; n_w < image_cols; ++n_w) {
      text_split_images += fake_token_around_image +
                          "<row_" + std::to_string(n_h + 1) + "_col_" + std::to_string(n_w + 1) + ">";
      // Repeat image_token image_seq_len times
      for (int i = 0; i < image_seq_len; ++i) {
        text_split_images += image_token;
      }
    }
    text_split_images += "\n";
  }
  
  // Add global image token section
  text_split_images += "\n" + fake_token_around_image + global_image_token;
  // Repeat image_token image_seq_len times
  for (int i = 0; i < image_seq_len; ++i) {
    text_split_images += image_token;
  }
  text_split_images += fake_token_around_image;
  
  return text_split_images;
}

// Function to generate expanded image tokens for a single image
std::string PromptSingleImage(
    int image_seq_len,
    const std::string& fake_token_around_image,
    const std::string& image_token,
    const std::string& global_image_token) {
  std::string result = fake_token_around_image + global_image_token;
  
  // Repeat image_token image_seq_len times
  for (int i = 0; i < image_seq_len; ++i) {
    result += image_token;
  }
  
  result += fake_token_around_image;
  
  return result;
}

// Function to process image tokens in the prompt
std::string ProcessImageTokens(const std::string& prompt) {
  std::string processed_prompt = prompt;
  
  // Check if <image> token exists in the prompt
  size_t image_pos = processed_prompt.find("<image>");
  if (image_pos != std::string::npos) {
    std::string expanded_image_content;
    
    // Choose between single image and split image mode based on flag
    if (absl::GetFlag(FLAGS_use_single_image)) {
      // Generate expanded image content for single image
      expanded_image_content = PromptSingleImage(
          64,  // image_seq_len
          "<fake_token_around_image>",  // fake_token_around_image
          "<image>",                    // image_token
          "<global-img>"                // global_image_token
      );
      printf("Found <image> token, replaced with single image expanded content\n");
    } else {
      // Generate expanded image content for split image (3x4 grid)
      expanded_image_content = PromptSplitImage(
          64,  // image_seq_len
          3,   // image_rows
          4,   // image_cols
          "<fake_token_around_image>",  // fake_token_around_image
          "<image>",                    // image_token
          "<global-img>"                // global_image_token
      );
      printf("Found <image> token, replaced with split image expanded content (3x4 grid)\n");
    }
    
    // Replace <image> with expanded content
    processed_prompt.replace(image_pos, 7, expanded_image_content);  // 7 is length of "<image>"
  }
  
  return processed_prompt;
}

// Function to process escape sequences in strings
std::string ProcessEscapeSequences(const std::string& input) {
  std::string result;
  result.reserve(input.length());
  
  for (size_t i = 0; i < input.length(); ++i) {
    if (input[i] == '\\' && i + 1 < input.length()) {
      switch (input[i + 1]) {
        case 'n':
          result += '\n';
          ++i;  // Skip the next character
          break;
        case 't':
          result += '\t';
          ++i;
          break;
        case 'r':
          result += '\r';
          ++i;
          break;
        case '\\':
          result += '\\';
          ++i;
          break;
        case '"':
          result += '"';
          ++i;
          break;
        default:
          result += input[i];
          break;
      }
    } else {
      result += input[i];
    }
  }
  
  return result;
}

std::unique_ptr<tflite::FlatBufferModel> LoadModel() {
  std::unique_ptr<tflite::FlatBufferModel> model =
      tflite::FlatBufferModel::BuildFromFile(
          absl::GetFlag(FLAGS_tflite_model).c_str());
  MINIMAL_CHECK(model != nullptr);
  return model;
}

void ApplyXNNPACKWeightCaching(tflite::Interpreter* interpreter) {
  auto delegate_options = TfLiteXNNPackDelegateOptionsDefault();
  std::string weight_cache_path = absl::GetFlag(FLAGS_weight_cache_path);
  delegate_options.weight_cache_file_path = weight_cache_path.c_str();
  delegate_options.num_threads = absl::GetFlag(FLAGS_num_threads);
  delegate_options.flags |=
      TFLITE_XNNPACK_DELEGATE_FLAG_ENABLE_SUBGRAPH_RESHAPING;
  delegate_options.flags |=
      TFLITE_XNNPACK_DELEGATE_FLAG_ENABLE_LATEST_OPERATORS;

  MINIMAL_CHECK(interpreter->ModifyGraphWithDelegate(
                    tflite::Interpreter::TfLiteDelegatePtr(
                        TfLiteXNNPackDelegateCreate(&delegate_options),
                        [](TfLiteDelegate* delegate) {
                          TfLiteXNNPackDelegateDelete(delegate);
                        })) == kTfLiteOk);
}

std::unique_ptr<tflite::Interpreter> BuildInterpreter(
    tflite::FlatBufferModel* model, int num_threads) {
  tflite::ops::builtin::BuiltinOpResolver resolver;
  // NOTE: We need to manually register optimized OPs for KV-cache and
  // Scaled Dot Product Attention (SDPA).
  tflite::ops::custom::GenAIOpsRegisterer(&resolver);
  tflite::InterpreterBuilder builder(*model, resolver);
  MINIMAL_CHECK(builder.SetNumThreads(num_threads) == kTfLiteOk);
  std::unique_ptr<tflite::Interpreter> interpreter;
  builder(&interpreter);
  MINIMAL_CHECK(interpreter != nullptr);

  if (!absl::GetFlag(FLAGS_weight_cache_path).empty()) {
    // optionally use xnnpack with weight caching
    ApplyXNNPACKWeightCaching(interpreter.get());
  }

  return interpreter;
}

std::map<std::string, std::vector<float, AlignedAllocator<float>>> BuildKVCache(
    tflite::Interpreter* interpreter) {
  tflite::SignatureRunner* runner = interpreter->GetSignatureRunner("decode");
  if (runner == nullptr) {
    return {};
  }
  // The arguments excluded are `tokens`, `input_pos`, and optionally `mask` and `pixel_values`.
  int excluded_inputs = 2;  // tokens and input_pos
  if (runner->input_tensor("mask") != nullptr) {
    excluded_inputs += 1;  // mask
  }
  if (runner->input_tensor("pixel_values") != nullptr) {
    excluded_inputs += 1;  // pixel_values
  }
  size_t num_layers = (runner->input_size() - excluded_inputs) / 2;
  if (num_layers == 0) {
    return {};
  }

  std::map<std::string, std::vector<float, AlignedAllocator<float>>> kv_cache;
  for (int i = 0; i < num_layers; ++i) {
    std::string k_cache_name = "kv_cache_k_" + std::to_string(i);
    std::string v_cache_name = "kv_cache_v_" + std::to_string(i);
    // We are assuming K and V tensors are of the same shape.
    TfLiteTensor* tensor = runner->input_tensor(k_cache_name.c_str());
    size_t count = tensor->bytes / sizeof(float);
    kv_cache.emplace(k_cache_name,
                     std::vector<float, AlignedAllocator<float>>(count, 0.0));
    kv_cache.emplace(v_cache_name,
                     std::vector<float, AlignedAllocator<float>>(count, 0.0));
  }

  return kv_cache;
}

void PrepareRunner(
    tflite::SignatureRunner* runner,
    std::map<std::string, std::vector<float, AlignedAllocator<float>>>&
        kv_cache) {
  for (auto& [name, cache] : kv_cache) {
    TfLiteCustomAllocation allocation = {
        .data = static_cast<void*>(cache.data()),
        .bytes = cache.size() * sizeof(float)};
    // Both input and output tensors are set to the same buffer. Not all
    // delegates support this in-place update. For those cases, we need to do
    // a ping-pong buffer and update the pointers between inference calls.
    MINIMAL_CHECK(runner->SetCustomAllocationForInputTensor(
                      name.c_str(), allocation) == kTfLiteOk);
    MINIMAL_CHECK(runner->SetCustomAllocationForOutputTensor(
                      name.c_str(), allocation) == kTfLiteOk);
  }
  MINIMAL_CHECK(runner->AllocateTensors() == kTfLiteOk);
}

tflite::SignatureRunner* GetPrefillRunner(
    tflite::Interpreter* interpreter, std::size_t num_input_tokens,
    std::map<std::string, std::vector<float, AlignedAllocator<float>>>&
        kv_cache,
    const LoRA* lora) {
  // Find the prefill signature length that best matches the input token size.
  // First, try to find prefill signatures with pixel support.
  tflite::SignatureRunner* runner = nullptr;
  int best_seq_size = -1;
  int delta = std::numeric_limits<int>::max();
  bool found_pixel_prefill = false;
  
  // First pass: Look for prefill signatures with pixel support (if enabled)
  if (absl::GetFlag(FLAGS_use_pixel_prefill)) {
    for (const std::string* key : interpreter->signature_keys()) {
      if (!absl::StrContains(*key, "prefill") ||
          absl::StrContains(*key, "lora") ||
          !absl::StrContains(*key, "pixel")) {
        continue;
      }
      TfLiteTensor* input_pos = interpreter->GetSignatureRunner(key->c_str())
                                    ->input_tensor("input_pos");
      // The expected shape for input position is [Seq].
      int seq_size = input_pos->dims->data[0];
      if (num_input_tokens <= seq_size && seq_size - num_input_tokens < delta) {
        if (lora == nullptr) {
          runner = interpreter->GetSignatureRunner(key->c_str());
        }
        best_seq_size = seq_size;
        delta = seq_size - num_input_tokens;
        found_pixel_prefill = true;
      }
    }
  }
  
  // Second pass: If no pixel prefill found, fall back to regular prefill
  if (!found_pixel_prefill) {
    delta = std::numeric_limits<int>::max();
    for (const std::string* key : interpreter->signature_keys()) {
      if (!absl::StrContains(*key, "prefill") ||
          absl::StrContains(*key, "lora") ||
          absl::StrContains(*key, "pixel")) {
        continue;
      }
      TfLiteTensor* input_pos = interpreter->GetSignatureRunner(key->c_str())
                                    ->input_tensor("input_pos");
      // The expected shape for input position is [Seq].
      int seq_size = input_pos->dims->data[0];
      if (num_input_tokens <= seq_size && seq_size - num_input_tokens < delta) {
        if (lora == nullptr) {
          runner = interpreter->GetSignatureRunner(key->c_str());
        }
        best_seq_size = seq_size;
        delta = seq_size - num_input_tokens;
      }
    }
  }
  
  if (lora != nullptr) {
    runner = lora->GetPrefillRunner(interpreter, best_seq_size);
  }
  MINIMAL_CHECK(runner != nullptr);
  
  // Debug output to show which type of prefill signature was selected
  if (found_pixel_prefill) {
    printf("Selected prefill signature with pixel support, seq_size: %d\n", best_seq_size);
  } else {
    printf("Selected regular prefill signature (no pixel support), seq_size: %d\n", best_seq_size);
  }
  
  PrepareRunner(runner, kv_cache);
  return runner;
}

tflite::SignatureRunner* GetDecodeRunner(
    tflite::Interpreter* interpreter,
    std::map<std::string, std::vector<float, AlignedAllocator<float>>>&
        kv_cache,
    LoRA* lora) {
  tflite::SignatureRunner* runner =
      lora == nullptr ? interpreter->GetSignatureRunner("decode")
                      : lora->GetDecodeRunner(interpreter);
  MINIMAL_CHECK(runner != nullptr);
  PrepareRunner(runner, kv_cache);
  return runner;
}

std::unique_ptr<sentencepiece::SentencePieceProcessor>
LoadSentencePieceProcessor() {
  std::ifstream input(absl::GetFlag(FLAGS_sentencepiece_model),
                      std::ios::binary);
  std::string serialized_proto = std::string(
      std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>());
  auto processor = std::make_unique<sentencepiece::SentencePieceProcessor>();
  MINIMAL_CHECK(processor->LoadFromSerializedProto(serialized_proto).ok());
  return processor;
}

// Generate mask for attention similar to Python _get_mask function
void GenerateMask(TfLiteTensor* mask_tensor, int k) {
  // Get mask tensor dimensions
  int batch = mask_tensor->dims->data[0];
  int dim1 = mask_tensor->dims->data[1]; 
  int seq_len = mask_tensor->dims->data[2];
  int kv_cache_size = mask_tensor->dims->data[3];
  
  // Fill mask with -inf initially (equivalent to np.ones(shape) * float("-inf"))
  float neg_inf = -std::numeric_limits<float>::infinity();
  std::fill_n(mask_tensor->data.f, batch * dim1 * seq_len * kv_cache_size, neg_inf);
  
  // Apply np.triu logic: zero out elements below k-th diagonal in last 2 dimensions
  // np.triu operates only on the last two dimensions (seq_len x kv_cache_size)
  for (int b = 0; b < batch; ++b) {
    for (int d = 0; d < dim1; ++d) {
      for (int i = 0; i < seq_len; ++i) {
        for (int j = 0; j < kv_cache_size; ++j) {
          // np.triu(mask, k) zeros elements where column < row + k
          // This allows attention from query i to keys 0 through i+k-1
          if (j < i + k) {
            int idx = b * (dim1 * seq_len * kv_cache_size) + 
                      d * (seq_len * kv_cache_size) + 
                      i * kv_cache_size + j;
            mask_tensor->data.f[idx] = 0.0f;
          }
          // Elements where j >= i + k remain as -inf (no attention allowed)
        }
      }
    }
  }
}

// Generate pixel values from real image or dummy values as fallback
void GeneratePixelValues(TfLiteTensor* pixel_values_tensor) {
  if (pixel_values_tensor == nullptr) {
    return;
  }
  
  std::string image_path = absl::GetFlag(FLAGS_image_path);
  
  // Try to load real image first
  if (!image_path.empty()) {
    if (LoadAndProcessImage(image_path, pixel_values_tensor)) {
      printf("Successfully loaded and processed real image from: %s\n", image_path.c_str());
      return;
    } else {
      printf("Failed to load image from: %s, falling back to dummy values\n", image_path.c_str());
    }
  }
  
  // Fallback to dummy values
  printf("Using dummy pixel values\n");
  
  // Calculate total number of elements
  size_t total_elements = 1;
  for (int i = 0; i < pixel_values_tensor->dims->size; ++i) {
    total_elements *= pixel_values_tensor->dims->data[i];
  }
  
  // Fill with random values in range [-1, 1]
  std::srand(42);  // Fixed seed for reproducibility
  for (size_t i = 0; i < total_elements; ++i) {
    // Generate random float in range [-1, 1]
    float random_value = (static_cast<float>(std::rand()) / RAND_MAX) * 2.0f - 1.0f;
    pixel_values_tensor->data.f[i] = random_value;
  }
  
  printf("Generated dummy pixel values with shape [");
  for (int i = 0; i < pixel_values_tensor->dims->size; ++i) {
    printf("%d", pixel_values_tensor->dims->data[i]);
    if (i < pixel_values_tensor->dims->size - 1) printf(", ");
  }
  printf("] in range [-1, 1]\n");
}

// A basic greedy sampler (equivalent to argmax).
int GreedySampler(const TfLiteTensor* logits) {
  float max_value = -std::numeric_limits<float>::infinity();
  int max_index = 0;
  // logits shape: [Batch, Seq, Vocab], Dtype: float
  for (int i = 0; i < logits->dims->data[2]; ++i) {
    if (logits->data.f[i] > max_value) {
      max_value = logits->data.f[i];
      max_index = i;
    }
  }
  return max_index;
}

}  // namespace

int main(int argc, char* argv[]) {
  absl::ParseCommandLine(argc, argv);

  // Prepare required components.
  std::unique_ptr<tflite::FlatBufferModel> model = LoadModel();
  std::unique_ptr<tflite::Interpreter> interpreter =
      BuildInterpreter(model.get(), absl::GetFlag(FLAGS_num_threads));
  std::unique_ptr<sentencepiece::SentencePieceProcessor> sp_processor =
      LoadSentencePieceProcessor();
  std::map<std::string, std::vector<float, AlignedAllocator<float>>> kv_cache =
      BuildKVCache(interpreter.get());
  MINIMAL_CHECK(!kv_cache.empty());

  std::unique_ptr<LoRA> lora = nullptr;
  if (!absl::GetFlag(FLAGS_lora_path).empty()) {
    lora = LoRA::FromFile(absl::GetFlag(FLAGS_lora_path));
    MINIMAL_CHECK(lora != nullptr);
  }

  // Tokenize the input prompt.
  std::string prompt = ProcessEscapeSequences(absl::GetFlag(FLAGS_prompt));
  printf("Original prompt: %s\n", prompt.c_str());
  
  // Process image tokens if present
  prompt = ProcessImageTokens(prompt);
  printf("Final prompt after image processing: %s\n", prompt.c_str());

  std::vector<int> prompt_tokens;
  MINIMAL_CHECK(sp_processor->Encode(prompt, &prompt_tokens).ok());

  std::string start_token = absl::GetFlag(FLAGS_start_token);
  if (!start_token.empty()) {
    prompt_tokens.insert(prompt_tokens.begin(),
                         sp_processor->PieceToId((start_token)));
  }
  std::string stop_token = absl::GetFlag(FLAGS_stop_token);
  int stop_token_id = -1;
  if (!stop_token.empty()) {
    stop_token_id = sp_processor->PieceToId((stop_token));
  }

  printf("prompt_tokens: [");
  for (size_t i = 0; i < prompt_tokens.size(); ++i) {
    printf("%d", prompt_tokens[i]);
    if (i < prompt_tokens.size() - 1) printf(", ");
  }
  printf("]\n");
  printf("stop_token_id: %d\n", stop_token_id);

  // prompt_tokens = {    1, 11126,    42,   216,    34,    27,    34,    45,    47, 49279,198,  9519,  9531,    42};
  // stop_token_id = 49279;
  // printf("prompt_tokens: [");
  // for (size_t i = 0; i < prompt_tokens.size(); ++i) {
  //   printf("%d", prompt_tokens[i]);
  //   if (i < prompt_tokens.size() - 1) printf(", ");
  // }
  // printf("]\n");

  // Get prefill and decode signature runners.
  std::size_t effective_prefill_token_size = prompt_tokens.size() - 1;
  tflite::SignatureRunner* prefill_runner = GetPrefillRunner(
      interpreter.get(), effective_prefill_token_size, kv_cache, lora.get());
  MINIMAL_CHECK(prefill_runner != nullptr);
  tflite::SignatureRunner* decode_runner =
      GetDecodeRunner(interpreter.get(), kv_cache, lora.get());
  MINIMAL_CHECK(decode_runner != nullptr);

  // Get Input Tensors for each of the runners.
  // Shape: [Batch, Seq], Dtype: int32
  TfLiteTensor* prefill_input = prefill_runner->input_tensor("tokens");
  // Shape: [Seq], Dtype: int32
  TfLiteTensor* prefill_input_pos = prefill_runner->input_tensor("input_pos");
  // Shape: [Batch, Seq], Dtype: int32
  TfLiteTensor* decode_input = decode_runner->input_tensor("tokens");
  // Shape: [Seq], Dtype: int32
  TfLiteTensor* decode_input_pos = decode_runner->input_tensor("input_pos");
  // shape: [Batch, kv_cache_max, num_query_groups, head_dim]
  TfLiteTensor* kv_cache_k_0 = decode_runner->input_tensor("kv_cache_k_0");
  // Shape: [Batch, N, 3, 512, 512], Dtype: float32 (N=1 for single image, N=13 for split image)
  TfLiteTensor* pixel_values = prefill_runner->input_tensor("pixel_values");

  int max_seq_size = prefill_input->dims->data[1];
  int kv_cache_max_size = kv_cache_k_0->dims->data[1];

  // Fill in the inputs (assuming one batch).
  // NOTE: We skip the last token and use that during decode.
  int prefill_seq_size =
      std::min(static_cast<int>(prompt_tokens.size()), max_seq_size);
  std::memset(prefill_input->data.i32, 0, prefill_input->bytes);
  std::memset(prefill_input_pos->data.i32, 0, prefill_input_pos->bytes);
  for (int i = 0; i < prefill_seq_size - 1; ++i) {
    prefill_input->data.i32[i] = prompt_tokens[i];
    prefill_input_pos->data.i32[i] = i;
  }
  
  // Handle mask for prefill if mask input exists
  TfLiteTensor* prefill_mask = prefill_runner->input_tensor("mask");
  if (prefill_mask != nullptr) {
    // For prefill, mask has shape [batch=1, 1, seq_len, kv_cache_size].
    // Python comment: "We want mask[0, 0, i, j] = 0 for j<=i and -inf otherwise"
    // With k=1: np.triu(mask, k=1) zeros elements where j < i+1, i.e., j <= i
    GenerateMask(prefill_mask, 1);
    printf("Generated prefill mask with shape [%d, %d, %d, %d]\n", 
           prefill_mask->dims->data[0], prefill_mask->dims->data[1],
           prefill_mask->dims->data[2], prefill_mask->dims->data[3]);
  }
  
  // Generate pixel values for image input if pixel_values tensor exists
  if (pixel_values != nullptr) {
    GeneratePixelValues(pixel_values);
    printf("Using prefill signature with pixel_values input\n");
  } else {
    printf("Using prefill signature without pixel_values input (text-only)\n");
  }
  
  MINIMAL_CHECK(prefill_runner->Invoke() == kTfLiteOk);

  // Decode until max kv-cache size or user defined step limit, whichever is
  // smaller.
  int max_tokens = absl::GetFlag(FLAGS_max_generated_tokens);
  if (max_tokens == -1) {
    max_tokens = absl::GetFlag(FLAGS_max_decode_steps);
  }
  int max_decode_steps = max_tokens == -1 ? kv_cache_max_size : max_tokens;
  int decode_steps =
      std::min(max_decode_steps, kv_cache_max_size - prefill_seq_size);
  MINIMAL_CHECK(decode_steps > 0);

  std::vector<int> output_tokens;
  output_tokens.reserve(decode_steps);
  int next_token = prompt_tokens[prefill_seq_size - 1];
  int next_position = prefill_seq_size - 1;
  
  // Check if decode runner has mask input
  TfLiteTensor* decode_mask = decode_runner->input_tensor("mask");
  
  for (int i = 0; i < decode_steps; ++i) {
    decode_input->data.i32[0] = next_token;
    decode_input_pos->data.i32[0] = next_position;
    
    // Handle mask for decode if mask input exists
    if (decode_mask != nullptr) {
      // For decode, mask has shape [batch=1, 1, 1, kv_cache_size].
      // Python comment: "We want mask[0, 0, 0, j] = 0 for j<=next_pos and -inf otherwise"
      // With k=next_position+1: np.triu(mask, k) zeros elements where j < 0+k, i.e., j <= next_position
      GenerateMask(decode_mask, next_position + 1);
    }
    
    MINIMAL_CHECK(decode_runner->Invoke() == kTfLiteOk);
    next_token = GreedySampler(decode_runner->output_tensor("logits"));
    output_tokens.push_back(next_token);
    next_position += 1;
    if (next_token == stop_token_id) {
      break;
    }
  }

  // Detokenize the generated output.
  std::string output_text;
  MINIMAL_CHECK(sp_processor->Decode(output_tokens, &output_text).ok());

  printf("Prompt:\n%s\nOutput text:\n%s\n", prompt.c_str(),
         output_text.c_str());

  return 0;
}