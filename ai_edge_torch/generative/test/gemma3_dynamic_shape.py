# Copyright 2024 The AI Edge Torch Authors.
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
"""Export a toy Gemma3 model with dynamic prefill length."""
import traceback

from absl import app
from absl import flags
import ai_edge_torch
from ai_edge_torch import fx_infra
from ai_edge_torch.generative.examples.gemma3 import gemma3
from ai_edge_torch.generative.layers import kv_cache as kv_utils
from ai_edge_torch.odml_torch.experimental import torch_tfl
import torch


_TFLITE_PATH = flags.DEFINE_string(
    "tflite_path",
    "/tmp/gemma3_dynamic_shape.tflite",
    "The tflite file path to export.",
)

_IS_DYNAMIC_SHAPE = flags.DEFINE_bool(
    "is_dynamic_shape",
    True,
    "Whether to export the model with dynamic shape.",
)


def _debug_assert_tensors_all_close(
    actual_tensor: torch.Tensor,
    expected_tensor: torch.Tensor,
    tensor_name: str,
    stage: str,
    atol=1e-5,
    rtol=1e-5,
):
  """Assert that two tensors are close and print debug info on failure."""
  are_close = torch.allclose(
      actual_tensor, expected_tensor, atol=atol, rtol=rtol
  )

  if are_close:
    print(
        f"torch.allclose for {tensor_name} after {stage} with atol={atol},"
        f" rtol={rtol}: True"
    )
    return

  print(
      f"\nDEBUG: torch.allclose for {tensor_name} FAILED after {stage}!"
      f" (atol={atol}, rtol={rtol})"
  )
  abs_diff = torch.abs(actual_tensor - expected_tensor)
  max_abs_diff = torch.max(abs_diff).item()
  print(f"  Max absolute difference: {max_abs_diff}")

  # Count elements differing by more than the tolerance
  num_significantly_differing_elements = torch.sum(abs_diff > atol).item()
  print(
      f"  Number of elements with abs_diff > {atol}:"
      f" {num_significantly_differing_elements} (out of"
      f" {actual_tensor.numel()})"
  )

  # Show an example of a differing element (where abs_diff is max)
  if actual_tensor.numel() > 0:
    _, max_abs_diff_flat_idx = torch.max(abs_diff.view(-1), dim=0)
    max_abs_diff_idx_multidim = torch.unravel_index(
        max_abs_diff_flat_idx, actual_tensor.shape
    )
    print(
        "  Example element with max absolute difference (at index"
        f" {max_abs_diff_idx_multidim}):"
    )
    print(
        f"    Actual value: {actual_tensor[max_abs_diff_idx_multidim].item()}"
    )
    print(
        "    Expected value:"
        f" {expected_tensor[max_abs_diff_idx_multidim].item()}"
    )
    print(f"    Difference: {abs_diff[max_abs_diff_idx_multidim].item()}")

  # Calculate relative difference carefully
  epsilon = 1e-12
  rel_diff = abs_diff / (torch.abs(expected_tensor.float()) + epsilon)
  max_rel_diff = torch.max(rel_diff).item()
  print(f"  Max relative difference: {max_rel_diff}")

  # Show an example of a differing element (where rel_diff is max)
  if rel_diff.numel() > 0:
    _, max_rel_diff_flat_idx = torch.max(rel_diff.view(-1), dim=0)
    max_rel_diff_idx_multidim = torch.unravel_index(
        max_rel_diff_flat_idx, actual_tensor.shape
    )
    print(
        "  Example element with max relative difference (at index"
        f" {max_rel_diff_idx_multidim}):"
    )
    print(
        f"    Actual value: {actual_tensor[max_rel_diff_idx_multidim].item()}"
    )
    print(
        "    Expected value:"
        f" {expected_tensor[max_rel_diff_idx_multidim].item()}"
    )

  assert (
      are_close
  ), f"{tensor_name} mismatch after {stage}. See DEBUG printouts for details."


class Gemma3Wrapper(torch.nn.Module):
  """A wrapper around the Gemma3 model."""

  def __init__(self, model: gemma3.Gemma3MM):
    super().__init__()
    self.model = model
    self.config = model.config

  def forward(
      self,
      tokens,
      input_pos,
      mask,
      all_k_caches: tuple[torch.Tensor],
      all_v_caches: tuple[torch.Tensor],
  ):
    # Reconstruct the KVCache object from flattened kv_cache arguments
    num_layers = self.model.config.num_layers
    caches = []
    for k_cache, v_cache in zip(all_k_caches, all_v_caches):
      caches.append(kv_utils.KVCacheEntry(k_cache, v_cache))
    reconstructed_kv_cache = kv_utils.KVCache(caches)

    # Call the original model's forward pass
    return self.model(
        tokens=tokens,
        input_pos=input_pos,
        mask=mask,
        kv_cache=reconstructed_kv_cache,
    )


class Gemma3Runner:
  """A runner for Gemma3 model."""

  def __init__(self, prefill_len=200, max_seq_len=2048, is_dynamic_shape=True):
    self.prefill_len = prefill_len
    self.max_seq_len = max_seq_len
    self.batch_size = 1
    self.is_dynamic_shape = is_dynamic_shape

    # Export a fake Gemma3 model with a dynamic prefill length.
    self.original_model = gemma3.build_model_1b(checkpoint_path=None)
    # Wrap the model
    self.model = Gemma3Wrapper(self.original_model)
    self.config = self.model.config
    self.vocab_size = self.config.vocab_size
    self.num_layers = self.config.num_layers

    self._create_inputs()

    if self.is_dynamic_shape:
      self._define_dynamic_shapes()
    else:
      self.dynamic_shapes = None

  def _create_inputs(self):
    torch.manual_seed(0)
    self.tokens = torch.randint(
        0, self.vocab_size, (self.batch_size, self.prefill_len), dtype=torch.int
    )
    self.input_pos = torch.arange(0, self.prefill_len, dtype=torch.int)

    # Create a causal mask for prefill.
    self.mask = torch.zeros(
        self.batch_size,
        1,
        self.prefill_len,
        self.max_seq_len,
        dtype=torch.float32,
    )
    causal_mask = torch.triu(
        torch.full((self.prefill_len, self.prefill_len), float("-inf")),
        diagonal=1,
    )
    self.mask[:, :, :, : self.prefill_len] = causal_mask

    # Create random KV caches.
    kv = kv_utils.KVCache.from_model_config(self.max_seq_len, self.config)
    all_k_caches_list = [torch.randn_like(entry.k_cache) for entry in kv.caches]
    all_v_caches_list = [torch.randn_like(entry.v_cache) for entry in kv.caches]

    # Prepare a flat dictionary of all model inputs.
    self.all_model_inputs_flat = {
        "tokens": self.tokens,
        "input_pos": self.input_pos,
        "mask": self.mask,
        "all_k_caches": tuple(all_k_caches_list),
        "all_v_caches": tuple(all_v_caches_list),
    }

  def _define_dynamic_shapes(self):
    # Create a dynamic sequence length.
    seq_len_dim = torch.export.Dim(
        "sequence_length", min=1, max=self.max_seq_len
    )
    # Create a dynamic sequence length for KV cache.
    kv_seq_len_dim = torch.export.Dim("kv_seq_len")

    # Define the dynamic shape spec for a single key/value tensor
    tensor_dynamic_spec = {1: kv_seq_len_dim}

    # Create tuples of the dynamic shapes for all K and V caches
    all_k_shapes = tuple(tensor_dynamic_spec for _ in range(self.num_layers))
    all_v_shapes = tuple(tensor_dynamic_spec for _ in range(self.num_layers))

    self.dynamic_shapes = {
        "tokens": {1: seq_len_dim},
        "input_pos": {0: seq_len_dim},
        "mask": {2: seq_len_dim, 3: kv_seq_len_dim},
        "all_k_caches": all_k_shapes,
        "all_v_caches": all_v_shapes,
    }

  def _export_model(self):
    self._run_original_model()

    print("Exporting model...")
    self.ep = torch.export.export(
        self.model,
        args=(),  # No positional arguments
        kwargs=self.all_model_inputs_flat,
        dynamic_shapes=self.dynamic_shapes,
    )

    self._run_exported_model("initial export")
    return self.ep

  def _run_original_model(self):
    print("running model()")
    self.original_model_output = self.model(**self.all_model_inputs_flat)
    self.original_logits = self.original_model_output["logits"]
    print(f"original_model_output logits shape: {self.original_logits.shape}")
    return self.original_model_output

  def _run_exported_model(self, stage: str):
    print(f"running ep.module() after {stage}")
    exported_model_output = self.ep.module()(**self.all_model_inputs_flat)
    exported_logits = exported_model_output["logits"]
    print(f"{stage} logits shape: {exported_logits.shape}")
    _debug_assert_tensors_all_close(
        exported_logits,
        self.original_logits,
        "logits",
        stage,
        atol=1e-3,
        rtol=1e-4,
    )
    return exported_model_output

  def _run_pre_lower_decompositions(self):
    try:
      print("Running TFL pre lower decompositions...")
      self.ep = fx_infra.safe_run_decompositions(
          self.ep, fx_infra.decomp.pre_lower_decomp()
      )
      print("TFL pre lower decompositions successful.")

      return self._run_exported_model("pre lower")
    except Exception as e:
      print(f"Error during pre lower decompositions: {e}")
      print(f"Traceback during pre lower ep.module: {traceback.format_exc()}")
      return None

  def _run_tfl_decompositions(self):
    try:
      print("Running TFL decompositions...")
      self.ep = self.ep.run_decompositions(torch_tfl.decomps)
      print("TFL decompositions successful.")

      return self._run_exported_model("TFL decompositions")
    except Exception as e:
      print(f"Error during TFL decompositions: {e}")
      print(f"Traceback during TFL decompositions: {traceback.format_exc()}")
      return None

  def convert_to_tflite(self, exported_path):
    self._export_model()

    if self.is_dynamic_shape:
      self._run_pre_lower_decompositions()
      self._run_tfl_decompositions()

    try:
      print("Running ai_edge_torch.convert...")
      self.edge_model = ai_edge_torch.convert(
          self.ep.module(),
          sample_args=(),
          sample_kwargs=self.all_model_inputs_flat,
          # dynamic_shapes=self.dynamic_shapes,
      )
      print("ai_edge_torch.convert successful.")

      self.edge_model.export(exported_path)
      print(f"Exported model to {exported_path}")

      self._run_tflite_model()
      return self.edge_model
    except Exception as e:
      print(f"Error during ai_edge_torch.convert: {e}")
      return None

  def _run_tflite_model(self):
    try:
      final_model_inputs = {
          "tokens": self.all_model_inputs_flat["tokens"],
          "input_pos": self.all_model_inputs_flat["input_pos"],
          "mask": self.all_model_inputs_flat["mask"],
      }
      for i, k_cache in enumerate(self.all_model_inputs_flat["all_k_caches"]):
        final_model_inputs[f"all_k_caches_{i}"] = k_cache
      for i, v_cache in enumerate(self.all_model_inputs_flat["all_v_caches"]):
        final_model_inputs[f"all_v_caches_{i}"] = v_cache

      edge_model_output = self.edge_model(**final_model_inputs)
      edge_logits = torch.from_numpy(edge_model_output["logits"])
      print(f"edge_model_output logits shape: {edge_logits.shape}")

      _debug_assert_tensors_all_close(
          edge_logits,
          self.original_logits,
          "logits",
          "converted model",
          atol=1e-3,
          rtol=1e-4,
      )
      return edge_model_output
    except Exception as e:
      print(f"Error during edge_model: {e}")
      print(f"Traceback during edge_model: {traceback.format_exc()}")
      return None


def main(_):
  runner = Gemma3Runner(is_dynamic_shape=_IS_DYNAMIC_SHAPE.value)
  runner.convert_to_tflite(_TFLITE_PATH.value)


if __name__ == "__main__":
  app.run(main)
