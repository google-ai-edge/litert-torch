# Gemma 3 Notes

The Gemma 3 is the latest model in the Gemma family of open weights models.

## Tokenizer

The Gemma 3 Tokenizer is available in the Gemma PyTorch repo [here](https://github.com/google/gemma_pytorch). The reauthored models here are compatible with that tokenizer.

## Convert & Quantize Gemma 3 to TFlite

Converting and quantizing a Gemma 3 model to various quantization schemes can be done using the following command:

```bash
python convert_gemma3_to_tflite.py \
 --quantize=<string for the desired quantization schemes> \
 --checkpoint_path=<path to torch safetensor directory> \
 --output_path=<path the directory where the tflite file to be saved> \
 --prefill_seq_lens=<maximum length of supported input> \
 --kv_cache_max_len=<maximum of prefill + decode context length> \
 --mask_as_input=True
```

where string for desired quantization scheme can be found in [`converter.py`](../utilities/converter.py#L46)

For example, the following command creates a Gemma3 model with dynamic int4 block32 quantization scheme:

```bash
python convert_gemma3_to_tflite.py --quantize="dynamic_int4_block32" \
 --checkpoint_path=/tmp/gemma-3-pytorch-gemma-3-1b-pt-v1 --output_path="/tmp/" \
 --prefill_seq_lens=2048 --kv_cache_max_len=4096 --mask_as_input=True
```

## Quality evaluation and performance benchmark

| Quantization Scheme  | Model Size | PIQA Score | CPU Prefill Speed | CPU Decode Speed | Peak Memory Usage|
| -------------------  | ---------- | ---------- | ----------------- | ---------------- |------------------|
| Dynamic INT8         | 973 MB     | 73.6       | 173 tokens/s      | 35 tokens/s      | 1.63 GB          |
| Dynamic INT4 Block32 | 711 MB     | 72.9       | 124 tokens/s      | 41 tokens/s      | 1.41 GB          |
| Dynamic INT4 Block128| 650 MB     | 71.6       | 146 tokens/s      | 43 tokens/s      | 1.31 GB          |

Note: All speed & memory values above are benchmarked on a Snapdragon 8 Elite device, performance may vary from device to device


## Gemma 3 Task File Creation

Creation of a Task file is needed to use the converted model and tokenizer in the [LLM Inference API](https://ai.google.dev/edge/mediapipe/solutions/genai/llm_inference). To create the `.task` file for Gemma 3, pip install the [mediapipe](https://pypi.org/project/mediapipe/) Python package and then execute the following Python code:

```python
TFLITE_MODEL = <path to your converted TF Lite model>
TOKENIZER_MODEL = <path to the Gemma 3 tokenizer from the Gemma Pytorch repo>
START_TOKEN="<bos>"
STOP_TOKENS=["<eos>", "<end_of_turn>"]
from mediapipe.tasks.python.genai import bundler
config = bundler.BundleConfig(
    tflite_model=TFLITE_MODEL,
    tokenizer_model=TOKENIZER_MODEL,
    start_token=START_TOKEN,
    stop_tokens=STOP_TOKENS,
    output_filename="/tmp/gemma3.task",
    prompt_prefix_user="<start_of_turn>user\n",
    prompt_suffix_user="<end_of_turn>\n<start_of_turn>model\n",
)
bundler.create_bundle(config)
```
