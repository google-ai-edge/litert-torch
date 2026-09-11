# AI Edge Examples

This module offers illustrations of how to utilize and run exported models with image support.

## Dependencies

You need to download [Android NDK](https://developer.android.com/ndk/downloads) (for example android-ndk-r18b) and set ANDROID_NDK env variable:  
```
export ANDROID_NDK=/mnt/c/ndk/android-ndk-r18b  
```

Install bazel:  
https://bazel.build/install/ubuntu



## Run inference:

```
mkdir cache

bazel run --verbose_failures -c opt //ai_edge_torch/generative/examples/cpp_image:text_generator_main -- \
--tflite_model="/home/dragynir/ai_vlm/ai-edge-torch-smalvlm/ai_edge_torch/generative/examples/smalvlm/models/SmolVLM-256M-Instruct-tflite-single/smalvlm-256m-instruct_q8_ekv2048.tflite" \
--sentencepiece_model="/home/dragynir/ai_vlm/ai-edge-torch-smalvlm/ai_edge_torch/generative/examples/smalvlm/models/SmolVLM-256M-Instruct-tflite/tokenizer.model" \
--start_token="<|im_start|>" --stop_token="<end_of_utterance>" --num_threads=16 \
--prompt="User:<image>What in the image?<end_of_utterance>\nAssistant:" --weight_cache_path="/home/dragynir/llm/ai-edge-torch/ai_edge_torch/generative/examples/cpp/cache/model.xnnpack_cache" --use_single_image=true --image_path="/home/dragynir/ai_vlm/car.jpg" --max_generated_tokens=64
```

Current implementation supports only single image input. Single image input model convertation:

```bash
# Convert for single image input
python convert_to_tflite.py --quantize="dynamic_int8"\
 --checkpoint_path='./models/SmolVLM-256M-Instruct' --output_path="./models/SmolVLM-256M-Instruct-tflite-single"\
 --mask_as_input=True --prefill_seq_lens=256 --kv_cache_max_len=2048 --do_image_splitting=False
```

Also, text_generator_main use simple image resizing that equivalent to this function:

```python
def create_custom_pixel_values(image: Image.Image) -> np.ndarray:
    """
    Create custom pixel_values tensor with shape [1, 1, 3, 512, 512].
    
    Args:
        image: PIL Image to process
        
    Returns:
        numpy array with shape [1, 1, 3, 512, 512], normalized with mean=[0.5, 0.5, 0.5] and std=[0.5, 0.5, 0.5]
    """
    # Resize image to 512x512
    resized_image = image.resize((512, 512), Image.BILINEAR)
    
    # Convert to numpy array and normalize to [0, 1]
    img_array = np.array(resized_image).astype(np.float32) / 255.0
    
    # Debug: Print first few raw pixel values (before normalization)
    print("Raw pixel values (first 10):", end=" ")
    raw_pixels = np.array(resized_image).flatten()[:10]
    for pixel in raw_pixels:
        print(f"{pixel}", end=" ")
    print()
    
    # Transpose from HWC to CHW format
    img_array = img_array.transpose(2, 0, 1)  # Shape: [3, 512, 512]
    
    # Normalize with mean=[0.5, 0.5, 0.5] and std=[0.5, 0.5, 0.5]
    # This is equivalent to (pixel - mean) / std
    img_array = (img_array - 0.5) / 0.5
    
    # Debug: Print first few normalized values
    print("Normalized values (first 10):", end=" ")
    normalized_flat = img_array.flatten()[:10]
    for value in normalized_flat:
        print(f"{value:.6f}", end=" ")
    print()
    
    # Add batch dimensions to get shape [1, 1, 3, 512, 512]
    img_array = img_array[np.newaxis, np.newaxis, ...]
    
    return torch.from_numpy(img_array)
```