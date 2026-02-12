from typing import Dict
from ai_edge_litert import interpreter as interpreter_lib
import numpy as np
import sys
from collections.abc import Sequence
from transformers import AutoProcessor
from PIL import Image
import requests


import torch
from transformers import AutoModelForVision2Seq


def _get_mask(shape: Sequence[int], k: int):
  """Gets the mask for the input to the model.

  Args:
  shape: The shape of the mask input to the model.
  k: all elements below the k-th diagonal are set to 0.

  Returns:
  The mask for the input to the model. All the elements in the mask are set
  to -inf except that all the elements below the k-th diagonal are set to 0.
  """
  mask = np.ones(shape, dtype=np.float32) * float("-inf")
  mask = np.triu(mask, k=k)
  return mask


class LiteRTLlmPipeline:

  def __init__(self, interpreter, processor):
    """Initializes the pipeline."""
    self._interpreter = interpreter
    self._processor = processor

    self._prefill_runner = None
    self._decode_runner = self._interpreter.get_signature_runner("decode")

  def _init_prefill_runner(self, num_input_tokens: int):
    """Initializes all the variables related to the prefill runner.

    This method initializes the following variables:
      - self._prefill_runner: The prefill runner based on the input size.
      - self._max_seq_len: The maximum sequence length supported by the model.

    Args:
      num_input_tokens: The number of input tokens.
    """
    if not self._interpreter:
      raise ValueError("Interpreter is not initialized.")

    # Prefill runner related variables will be initialized in `predict_text` and
    # `compute_log_likelihood`.
    self._prefill_runner = self._get_prefill_runner(num_input_tokens)
    # input_token_shape has shape (batch, max_seq_len)
    input_token_shape = self._prefill_runner.get_input_details()["tokens"][
        "shape"
    ]
    if len(input_token_shape) == 1:
      self._max_seq_len = input_token_shape[0]
    else:
      self._max_seq_len = input_token_shape[1]

    # SmolLM: kv cache input has shape [batch=1, cache_size, num_kv_heads, head_dim].
    kv_cache_shape = self._prefill_runner.get_input_details()["kv_cache_k_0"][
        "shape"
    ]
    self._max_kv_cache_seq_len = kv_cache_shape[1]

  def _init_kv_cache(self) -> dict[str, np.ndarray]:
    if self._prefill_runner is None:
      raise ValueError("Prefill runner is not initialized.")
    kv_cache = {}
    for input_key in self._prefill_runner.get_input_details().keys():
      if "kv_cache" in input_key:
        kv_cache[input_key] = np.zeros(
            self._prefill_runner.get_input_details()[input_key]["shape"],
            dtype=np.float32,
        )
        kv_cache[input_key] = np.zeros(
            self._prefill_runner.get_input_details()[input_key]["shape"],
            dtype=np.float32,
        )
    return kv_cache

  def _get_prefill_runner(self, num_input_tokens: int):
    """Gets the prefill runner with the best suitable input size.

    Args:
      num_input_tokens: The number of input tokens.

    Returns:
      The prefill runner with the smallest input size.
    """
    best_signature = None
    delta = sys.maxsize
    max_prefill_len = -1
    for key in self._interpreter.get_signature_list().keys():
      if "prefill" not in key or "pixel" not in key:
        continue
      input_pos = self._interpreter.get_signature_runner(
          key
      ).get_input_details()["input_pos"]
      # input_pos["shape"] has shape (max_seq_len, )
      seq_size = input_pos["shape"][0]
      max_prefill_len = max(max_prefill_len, seq_size)
      if num_input_tokens <= seq_size and seq_size - num_input_tokens < delta:
        delta = seq_size - num_input_tokens
        best_signature = key
    if best_signature is None:
      raise ValueError(
          "The largest prefill length supported is %d, but we have %d number of"
          " input tokens" % (max_prefill_len, num_input_tokens)
      )
    return self._interpreter.get_signature_runner(best_signature)

  def _run_prefill(
      self,
      prefill_token_ids: Sequence[int],
      pixel_values: np.ndarray,
  ) -> dict[str, np.ndarray]:
    """Runs prefill and returns the kv cache.

    Args:
      prefill_token_ids: The token ids of the prefill input.

    Returns:
      The updated kv cache.
    """
    if not self._prefill_runner:
      raise ValueError("Prefill runner is not initialized.")
    prefill_token_length = len(prefill_token_ids)
    if prefill_token_length == 0:
      return self._init_kv_cache()

    # Prepare the input to be [1, max_seq_len].
    input_token_ids = [0] * self._max_seq_len
    input_token_ids[:prefill_token_length] = prefill_token_ids
    input_token_ids = np.asarray(input_token_ids, dtype=np.int32)
    input_token_ids = np.expand_dims(input_token_ids, axis=0)

    # Prepare the input position to be [max_seq_len].
    input_pos = [0] * self._max_seq_len
    input_pos[:prefill_token_length] = range(prefill_token_length)
    input_pos = np.asarray(input_pos, dtype=np.int32)

    # Initialize kv cache.
    prefill_inputs = self._init_kv_cache()
    # Prepare the tokens and input position inputs.
    prefill_inputs.update({
        "tokens": input_token_ids,
        "input_pos": input_pos,
        "pixel_values": pixel_values,
    })
    if "mask" in self._prefill_runner.get_input_details().keys():
      # For prefill, mask has shape [batch=1, 1, seq_len, kv_cache_size].
      # We want mask[0, 0, i, j] = 0 for j<=i and -inf otherwise.
      prefill_inputs["mask"] = _get_mask(
          shape=self._prefill_runner.get_input_details()["mask"]["shape"],
          k=1,
      )
    prefill_outputs = self._prefill_runner(**prefill_inputs)
    if "logits" in prefill_outputs:
      # Prefill outputs includes logits and kv cache. We only output kv cache.
      prefill_outputs.pop("logits")

    return prefill_outputs

  def _greedy_sampler(self, logits: np.ndarray) -> int:
    return int(np.argmax(logits))

  def _run_decode(
      self,
      start_pos: int,
      start_token_id: int,
      kv_cache: dict[str, np.ndarray],
      max_decode_steps: int,
  ) -> str:
    """Runs decode and outputs the token ids from greedy sampler.

    Args:
      start_pos: The position of the first token of the decode input.
      start_token_id: The token id of the first token of the decode input.
      kv_cache: The kv cache from the prefill.
      max_decode_steps: The max decode steps.

    Returns:
      The token ids from the greedy sampler.
    """
    next_pos = start_pos
    next_token = start_token_id
    decode_text = []
    decode_inputs = kv_cache

    for _ in range(max_decode_steps):
      decode_inputs.update({
          "tokens": np.array([[next_token]], dtype=np.int32),
          "input_pos": np.array([next_pos], dtype=np.int32),
      })
      if "mask" in self._decode_runner.get_input_details().keys():
        # For decode, mask has shape [batch=1, 1, 1, kv_cache_size].
        # We want mask[0, 0, 0, j] = 0 for j<=next_pos and -inf otherwise.
        decode_inputs["mask"] = _get_mask(
            shape=self._decode_runner.get_input_details()["mask"]["shape"],
            k=next_pos + 1,
        )
      decode_outputs = self._decode_runner(**decode_inputs)
      # Output logits has shape (batch=1, 1, vocab_size). We only take the first
      # element.
      logits = decode_outputs.pop("logits")[0][0]
      next_token = self._greedy_sampler(logits)
      if next_token == self._processor.tokenizer.eos_token_id:
        break
      decode_text.append(
          self._processor.tokenizer.decode(next_token, skip_special_tokens=True)
      )
      if len(decode_text[-1]) == 0:
        # Break out the loop if we hit the special token.
        break

      print(decode_text[-1], end="", flush=True)
      # Decode outputs includes logits and kv cache. We already poped out
      # logits, so the rest is kv cache. We pass the updated kv cache as input
      # to the next decode step.
      decode_inputs = decode_outputs
      next_pos += 1

    print()  # print a new line at the end.
    return "".join(decode_text)

  def generate(self, inputs: Dict, max_decode_steps: int | None = None) -> str:

    token_ids = inputs["input_ids"][0]
    pixel_values = inputs["pixel_values"][0]

    # Initialize the prefill runner with the suitable input size.
    self._init_prefill_runner(len(token_ids))

    # Run prefill.
    # Prefill up to the seond to the last token of the prompt, because the last
    # token of the prompt will be used to bootstrap decode.
    prefill_token_length = len(token_ids) - 1

    print("Running prefill")
    kv_cache = self._run_prefill(token_ids[:prefill_token_length], pixel_values)
    # Run decode.
    print("Running decode")
    actual_max_decode_steps = (
        self._max_kv_cache_seq_len - prefill_token_length - 1
    )
    if max_decode_steps is not None:
      actual_max_decode_steps = min(actual_max_decode_steps, max_decode_steps)
    decode_text = self._run_decode(
        prefill_token_length,
        token_ids[prefill_token_length],
        kv_cache,
        actual_max_decode_steps,
    )
    return decode_text


if __name__ == "__main__":

  model_id = "./models/SmolVLM-256M-Instruct"
  tflite_model_path = "./models/SmolVLM-256M-Instruct-tflite/smalvlm-256m-instruct_q8_ekv2048.tflite"

  interpreter = interpreter_lib.InterpreterWithCustomOps(
      custom_op_registerers=["pywrap_genai_ops.GenAIOpsRegisterer"],
      model_path=tflite_model_path,
      num_threads=2,
      experimental_default_delegate_latest_features=True,
  )

  processor = AutoProcessor.from_pretrained(model_id, do_image_splitting=True)
  image_url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/car.jpg?download=true"
  image = Image.open(requests.get(image_url, stream=True).raw)
  # image = Image.open("/home/dragynir/ai_vlm/cats.jpg")
  # image = Image.open("/home/dragynir/ai_vlm/car.jpg")

  messages = [
      {
          "role": "user",
          "content": [
              {"type": "image"},
              {"type": "text", "text": "What in the image?"},
          ],
      },
  ]
  prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
  print(prompt)
  inputs = processor(text=prompt, images=[image], return_tensors="pt")

  # Tflite model inference
  pipeline = LiteRTLlmPipeline(interpreter, processor)
  tflite_text = pipeline.generate(inputs, max_decode_steps=100)

  # HuggingFace model inference
  DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
  inputs = inputs.to(DEVICE)
  model = AutoModelForVision2Seq.from_pretrained(
      model_id,
      torch_dtype=torch.bfloat16,
      _attn_implementation="eager",
  ).to(DEVICE)
  generated_ids = model.generate(**inputs, do_sample=False, max_new_tokens=100)
  generated_texts = processor.batch_decode(
      generated_ids,
      skip_special_tokens=True,
  )

  hf_text = generated_texts[0]
  print("-" * 100)
  print("Tflite:\n", tflite_text)
  print("HF:\n", hf_text)
