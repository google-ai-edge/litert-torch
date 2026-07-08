# Calibration Dataset Reference Templates

This directory contains template dataset files to guide you in formatting your custom calibration datasets.

Statically quantizing a model requires profiling the range statistics of model activations against a small calibration dataset. The quality of your calibration dataset directly impacts the output quality of the quantized model.

We recommend using **between 20 to 100 sample prompts** that are representative of your production workload.

---

## Supported Formats

The calibration tools natively support two schema structures (using either `.json` or `.jsonl` extensions):

### 1. Simple Flat Prompt Format (`inputs` key)
Use this format if you have a list of raw prompt strings that do not require any chat templating. Each example is a dictionary containing a single `"inputs"` key.

*   **JSON Example ([sample_inputs.json](sample_inputs.json))**:
    ```json
    [
      { "inputs": "What is the capital of France?" },
      { "inputs": "Write a python program to add 2 and 3." }
    ]
    ```
*   **JSON Lines Example ([sample_inputs.jsonl](sample_inputs.jsonl))**:
    ```jsonl
    {"inputs": "What is the capital of France?"}
    {"inputs": "Write a python program to add 2 and 3."}
    ```

### 2. Conversational Chat Format (`messages` key)
Use this format if your model expects formatted chat dialog (e.g. instruction-tuned models with user/assistant roles). Each example contains a list of message objects under the `"messages"` key.

    *   If a HuggingFace tokenizer directory is supplied (`--transformers_model_path`),
        the calibration script automatically applies `.apply_chat_template()` to user
        messages. This ensures the correct system instructions, turn wraps, and
        formatting tokens are added to match the training context.
    *   If a SentencePiece tokenizer path (`--spm_path`) is used, it falls back to
        joining user messages with newlines.



*   **JSON Example ([sample_messages.json](sample_messages.json))**:
    ```json
    [
      {
        "messages": [
          { "role": "user", "content": "What is the capital of France?" }
        ]
      }
    ]
    ```
*   **JSON Lines Example ([sample_messages.jsonl](sample_messages.jsonl))**:
    ```jsonl
    {"messages": [{"role": "user", "content": "What is the capital of France?"}]}
    ```

---

## JSON vs JSONL (JSON Lines)

*   **JSON** is standard and easy to write for small hand-crafted datasets.
*   **JSONL** is highly recommended for larger datasets. It stores each entry on a separate line, allowing the python calibrator to stream and parse the file line-by-line instead of loading a massive JSON array structure entirely into RAM.

---

## Code Reference
The parsing logic for these schemas resides in [quant_utils.py](../quant_utils.py) under `get_example_prompt()`.
