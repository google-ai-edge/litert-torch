# Embedding Gemma-300M

[Embedding Gemma](https://huggingface.co/google/embeddinggemma-300m) is a text embedding model based on the Gemma architecture. This example demonstrates how to reauthor the model using the LiteRT Torch Generative API and convert it to TFLite.

## Model Details

EmbeddingGemma-300M is an encoder-only model with 24 layers. It uses a combination of local sliding window attention and global attention. This implementation supports:
- Sliding window attention mask.
- Mean pooling of hidden states.
- Final dense projections and L2 normalization (optional, useful for Matryoshka embeddings).

## Requirements

To run this example and verify the results, you need the following packages:

```bash
pip install litert-torch transformers sentence-transformers safetensors
```

## Convert to TFLite

To convert the model to TFLite, use the `convert_to_tflite.py` script. You'll need the HuggingFace checkpoint for `google/embeddinggemma-300m`.

```bash
python convert_to_tflite.py 
  --checkpoint_path=<path_to_checkpoint> 
  --output_path=/tmp/ 
  --quantize=dynamic_int8 
  --prefill_seq_lens=512 
  --final_l2_norm=True
```

### Conversion Flags

- `--checkpoint_path`: Path to the directory containing the model's `model.safetensors` and Dense projection layers.
- `--output_path`: Directory where the converted `.tflite` model will be saved.
- `--quantize`: Quantization scheme (e.g., `dynamic_int8`, `none`).
- `--prefill_seq_lens`: Defines the input sequence length for the converted TFLite model.
- `--final_l2_norm`: Whether to apply final L2 normalization to the embeddings. Defaults to `True`. Set to `False` if using Matryoshka embeddings.

## Verify the Model

You can verify the reauthored model's output against the original HuggingFace model using `verify.py`.

```bash
python verify.py 
  --checkpoint=<path_to_checkpoint> 
  --prompts="What is the meaning of life?" 
  --prompts="This is an example sentence."
```

The verification script compares the final embeddings produced by the original `sentence-transformers` implementation and the reauthored `litert_torch` implementation to ensure parity.