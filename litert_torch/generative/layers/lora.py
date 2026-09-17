# Copyright 2025 The LiteRT Torch Authors.
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

"""LoRA weights for generative models.

The current implementation support attention only lora. Additionally, we expect
lora weights for all projections within the attention module (i.e., Q, K, V, O).
"""

import dataclasses
from typing import Any, Callable, List, Optional, Tuple

from litert_torch.generative.layers import model_config
import flatbuffers
import numpy as np
import safetensors
import torch
import torch.utils._pytree as pytree

from tensorflow.lite.python import schema_py_generated as schema_fb  # pylint: disable=g-direct-tensorflow-import

_TFLITE_SCHEMA_VERSION = 3
_TFLITE_FILE_IDENTIFIER = b"TFL3"


@dataclasses.dataclass
class LoRAWeight:
  """LoRA weight per projection. The weights are pre-transposed."""

  a_prime: torch.Tensor
  b_prime: torch.Tensor

  def __eq__(self, other: Any, rtol: float = 1e-5, atol: float = 1e-8) -> bool:
    if not isinstance(other, LoRAWeight):
      return False
    if self.a_prime.shape != other.a_prime.shape:
      return False
    if self.b_prime.shape != other.b_prime.shape:
      return False
    return torch.allclose(
        self.a_prime, other.a_prime, rtol=rtol, atol=atol
    ) and torch.allclose(self.b_prime, other.b_prime, rtol=rtol, atol=atol)


@dataclasses.dataclass
class AttentionLoRA:
  """LoRA weights for attention module."""

  query: LoRAWeight
  key: LoRAWeight
  value: LoRAWeight
  output: LoRAWeight

  def __eq__(self, other: Any, rtol: float = 1e-5, atol: float = 1e-8) -> bool:
    if not isinstance(other, AttentionLoRA):
      return False
    return (
        self.query.__eq__(other.query, rtol=rtol, atol=atol)
        and self.key.__eq__(other.key, rtol=rtol, atol=atol)
        and self.value.__eq__(other.value, rtol=rtol, atol=atol)
        and self.output.__eq__(other.output, rtol=rtol, atol=atol)
    )


@dataclasses.dataclass
class LoRAEntry:
  """LoRA weights for a single layer."""

  attention: AttentionLoRA

  def __eq__(self, other: Any, rtol: float = 1e-5, atol: float = 1e-8) -> bool:
    if not isinstance(other, LoRAEntry):
      return False
    return self.attention.__eq__(other.attention, rtol=rtol, atol=atol)


@dataclasses.dataclass
class LoRATensorNames:
  """Tensor names for LoRA weights."""

  attn_query_w_a: str
  attn_query_w_b: str

  attn_key_w_a: str
  attn_key_w_b: str

  attn_value_w_a: str
  attn_value_w_b: str

  attn_output_w_a: str
  attn_output_w_b: str


@dataclasses.dataclass
class LoRA:
  """LoRA weights for all modules."""

  adapters: Tuple[LoRAEntry, ...]

  def __eq__(self, other: Any, rtol: float = 1e-5, atol: float = 1e-8) -> bool:
    if not isinstance(other, LoRA):
      return False
    if len(self.adapters) != len(other.adapters):
      return False
    return all(
        adapter.__eq__(other_adapter, rtol=rtol, atol=atol)
        for adapter, other_adapter in zip(self.adapters, other.adapters)
    )

  def get_rank(self) -> int:
    """Returns the rank of the LoRA weights."""
    return self.adapters[0].attention.query.a_prime.shape[1]

  @classmethod
  def from_safetensors(
      cls,
      path: str,
      scale: float,
      config: model_config.ModelConfig,
      lora_tensor_names: LoRATensorNames,
      dtype: torch.dtype = torch.float32,
  ) -> "LoRA":
    """Creates LoRA weights from a Hugging Face model.

    Args:
      path: Path to the model.
      scale: Scale factor for the LoRA weights (applied only to one of the
        projections). The scaling factor depnds on the training configuration.
        The common values are either `lora_alpha / rank` or `lora_alpha /
        sqrt(rank)`.
      config: Model configuration.
      lora_tensor_names: Tensor names for the LoRA weights.
      dtype: Data type of the LoRA weights. Currently only float32 is supported.

    Returns:
      LoRA weights for all modules.
    """
    with safetensors.safe_open(path, framework="pt", device="cpu") as f:
      adapters = []
      for i in range(config.num_layers):
        attention_lora = AttentionLoRA(
            query=LoRAWeight(
                a_prime=f.get_tensor(lora_tensor_names.attn_query_w_a.format(i))
                .to(dtype)
                .T
                * scale,
                b_prime=f.get_tensor(lora_tensor_names.attn_query_w_b.format(i))
                .to(dtype)
                .T,
            ),
            key=LoRAWeight(
                a_prime=f.get_tensor(lora_tensor_names.attn_key_w_a.format(i))
                .to(dtype)
                .T
                * scale,
                b_prime=f.get_tensor(lora_tensor_names.attn_key_w_b.format(i))
                .to(dtype)
                .T,
            ),
            value=LoRAWeight(
                a_prime=f.get_tensor(lora_tensor_names.attn_value_w_a.format(i))
                .to(dtype)
                .T
                * scale,
                b_prime=f.get_tensor(lora_tensor_names.attn_value_w_b.format(i))
                .to(dtype)
                .T,
            ),
            output=LoRAWeight(
                a_prime=f.get_tensor(
                    lora_tensor_names.attn_output_w_a.format(i)
                )
                .to(dtype)
                .T
                * scale,
                b_prime=f.get_tensor(
                    lora_tensor_names.attn_output_w_b.format(i)
                )
                .to(dtype)
                .T,
            ),
        )
        adapters.append(LoRAEntry(attention=attention_lora))
    return cls(adapters=adapters)

  @classmethod
  def from_flatbuffers(
      cls,
      flatbuffer_model: bytearray,
      dtype: torch.dtype = torch.float32,
  ) -> "LoRA":
    """Creates LoRA weights from FlatBuffers.

    Args:
      flatbuffer_model: FlatBuffers model.
      dtype: Data type of the LoRA weights.

    Returns:
      LoRA weights for all modules.
    """
    model = schema_fb.Model.GetRootAsModel(flatbuffer_model, 0)
    model = schema_fb.ModelT.InitFromObj(model)

    flat_names = []
    tensors = []
    for tensor in model.subgraphs[0].tensors:
      name = tensor.name.decode("utf-8")
      assert name.startswith("lora_")
      flat_names.append(name.split("lora_")[-1])
      buffer_bytes = model.buffers[tensor.buffer].data.data.tobytes()
      arr = np.frombuffer(buffer_bytes, dtype=np.float32).reshape(tensor.shape)
      torch_tensor = torch.from_numpy(arr).to(dtype)
      tensors.append(torch_tensor)

    return _unflatten_lora(tensors, (flat_names, []))

  @classmethod
  def zeros(
      cls,
      rank: int,
      config: model_config.ModelConfig,
      dtype: torch.dtype = torch.float32,
  ) -> "LoRA":
    """Creates LoRA weights with zeros.

    Args:
      rank: Rank of the LoRA weights.
      config: Model configuration.
      dtype: Data type of the LoRA weights. Currently only float32 is supported.

    Returns:
      LoRA weights with zeros.
    """
    return cls._from_tensor_generator(
        tensor_generator=lambda shape, dtype: torch.zeros(shape, dtype=dtype),
        rank=rank,
        config=config,
        dtype=dtype,
    )

  @classmethod
  def random(
      cls,
      rank: int,
      config: model_config.ModelConfig,
      dtype: torch.dtype = torch.float32,
  ) -> "LoRA":
    """Creates LoRA weights with random values.

    Args:
      rank: Rank of the LoRA weights.
      config: Model configuration.
      dtype: Data type of the LoRA weights.

    Returns:
      LoRA weights with random values.
    """
    return cls._from_tensor_generator(
        tensor_generator=lambda shape, dtype: torch.randint(
            low=0, high=128, size=shape, dtype=dtype
        ),
        rank=rank,
        config=config,
        dtype=dtype,
    )

  @classmethod
  def _from_tensor_generator(
      cls,
      tensor_generator: Callable[[Tuple[int, ...], torch.dtype], torch.Tensor],
      rank: int,
      config: model_config.ModelConfig,
      dtype: torch.dtype = torch.float32,
  ) -> "LoRA":
    """Creates LoRA weights from a tensor generator."""
    adapters = []

    for i in range(config.num_layers):
      block_config = config.block_config(i)
      q_per_kv = (
          block_config.attn_config.num_heads
          // block_config.attn_config.num_query_groups
      )
      q_out_dim = (
          block_config.attn_config.num_heads
          * block_config.attn_config.head_dim
      )
      k_out_dim = v_out_dim = (
          block_config.attn_config.num_query_groups
          * block_config.attn_config.head_dim
      )
      attention_lora = AttentionLoRA(
          query=LoRAWeight(
              a_prime=tensor_generator((config.embedding_dim, rank), dtype),
              b_prime=tensor_generator((rank, q_out_dim), dtype),
          ),
          key=LoRAWeight(
              a_prime=tensor_generator((config.embedding_dim, rank), dtype),
              b_prime=tensor_generator((rank, k_out_dim), dtype),
          ),
          value=LoRAWeight(
              a_prime=tensor_generator((config.embedding_dim, rank), dtype),
              b_prime=tensor_generator((rank, v_out_dim), dtype),
          ),
          output=LoRAWeight(
              a_prime=tensor_generator(
                  (
                      block_config.attn_config.num_heads
                      * block_config.attn_config.head_dim,
                      rank,
                  ),
                  dtype,
              ),
              b_prime=tensor_generator((rank, config.embedding_dim), dtype),
          ),
      )
      adapters.append(LoRAEntry(attention=attention_lora))
    return cls(adapters=adapters)

  def to_tflite(self) -> bytearray:
    """Converts LoRA to FlatBuffers."""
    return _lora_to_flatbuffers(self)


def apply_lora(
    x: torch.Tensor,
    lora_weight: LoRAWeight,
    shape: Optional[Tuple[int, ...]] = None,
) -> torch.Tensor:
  """Applies LoRA weights to a tensor.

  Args:
    x: Input tensor.
    lora_weight: LoRA weight.
    shape: Output shape. If None, the output shape is the same as the input
      shape.

  Returns:
    Output tensor.
  """
  output = torch.matmul(
      torch.matmul(x, lora_weight.a_prime), lora_weight.b_prime
  )
  if shape is not None:
    output = output.reshape(shape)
  return output


def _flatten_attention_lora(
    lora: AttentionLoRA, block_index: int
) -> Tuple[List[torch.Tensor], List[str]]:
  """Flattens LoRA weights for attention module."""
  flattened = []
  flat_names = []
  flattened.append(lora.query.a_prime)
  flat_names.append(f"atten_q_a_prime_weight_{block_index}")
  flattened.append(lora.query.b_prime)
  flat_names.append(f"atten_q_b_prime_weight_{block_index}")
  flattened.append(lora.key.a_prime)
  flat_names.append(f"atten_k_a_prime_weight_{block_index}")
  flattened.append(lora.key.b_prime)
  flat_names.append(f"atten_k_b_prime_weight_{block_index}")
  flattened.append(lora.value.a_prime)
  flat_names.append(f"atten_v_a_prime_weight_{block_index}")
  flattened.append(lora.value.b_prime)
  flat_names.append(f"atten_v_b_prime_weight_{block_index}")
  flattened.append(lora.output.a_prime)
  flat_names.append(f"atten_o_a_prime_weight_{block_index}")
  flattened.append(lora.output.b_prime)
  flat_names.append(f"atten_o_b_prime_weight_{block_index}")
  return flattened, flat_names


def _flatten_lora(lora: LoRA) -> Tuple[List[torch.Tensor], List[Any]]:
  """Flattens LoRA weights."""
  flattened = []
  flat_names = []
  none_names = []
  for i, entry in enumerate(lora.adapters):
    attn_flattened, attn_flat_names = _flatten_attention_lora(
        lora=entry.attention, block_index=i
    )
    flattened.extend(attn_flattened)
    flat_names.extend(attn_flat_names)
  return flattened, [flat_names, none_names]


def _flatten_lora_with_keys(lora: LoRA) -> Tuple[List[Any], List[Any]]:
  """Flattens LoRA weights with keys."""
  flattened, (flat_names, _) = _flatten_lora(lora)
  return [
      (pytree.MappingKey(k), v) for k, v in zip(flat_names, flattened)
  ], flat_names


def _unflatten_lora(
    values: List[torch.Tensor], context: Tuple[List[str], List[Any]]
) -> LoRA:
  """Unflattens LoRA object."""
  flat_names, _ = context
  names_weights = list(zip(flat_names, values))
  adapters = {}
  while names_weights:
    name, weight = names_weights.pop(0)
    block_idx = int(name.split("_")[-1])
    if block_idx not in adapters:
      adapters[block_idx] = LoRAEntry(
          attention=AttentionLoRA(
              query=LoRAWeight(
                  a_prime=None,
                  b_prime=None,
              ),
              key=LoRAWeight(
                  a_prime=None,
                  b_prime=None,
              ),
              value=LoRAWeight(
                  a_prime=None,
                  b_prime=None,
              ),
              output=LoRAWeight(
                  a_prime=None,
                  b_prime=None,
              ),
          )
      )

    if name.startswith("atten_"):
      if "q_a_prime" in name:
        adapters[block_idx].attention.query.a_prime = weight
      elif "q_b_prime" in name:
        adapters[block_idx].attention.query.b_prime = weight
      elif "k_a_prime" in name:
        adapters[block_idx].attention.key.a_prime = weight
      elif "k_b_prime" in name:
        adapters[block_idx].attention.key.b_prime = weight
      elif "v_a_prime" in name:
        adapters[block_idx].attention.value.a_prime = weight
      elif "v_b_prime" in name:
        adapters[block_idx].attention.value.b_prime = weight
      elif "o_a_prime" in name:
        adapters[block_idx].attention.output.a_prime = weight
      elif "o_b_prime" in name:
        adapters[block_idx].attention.output.b_prime = weight
      else:
        raise ValueError(f"Unsupported name: {name}")
    else:
      raise ValueError(f"Unsupported name: {name}")

  return LoRA(adapters=tuple(adapters[key] for key in sorted(adapters)))


pytree.register_pytree_node(
    LoRA,
    _flatten_lora,
    _unflatten_lora,
    flatten_with_keys_fn=_flatten_lora_with_keys,
    serialized_type_name="",
)


def _add_buffer(builder: flatbuffers.Builder, data: np.ndarray | None) -> int:
  """Adds a buffer to the FlatBuffers."""
  if data is not None:
    assert data.dtype == np.float32
    schema_fb.BufferStartDataVector(builder, data.size * data.itemsize)
    for value in reversed(data.flatten().tolist()):
      builder.PrependFloat32(value)
    data_offset = builder.EndVector()
  else:
    schema_fb.BufferStartDataVector(builder, 0)
    data_offset = builder.EndVector()

  schema_fb.BufferStart(builder)
  schema_fb.BufferAddData(builder, data_offset)
  buffer_offset = schema_fb.BufferEnd(builder)
  return buffer_offset


def _add_tensor(
    builder: flatbuffers.Builder,
    name: str,
    shape: Tuple[int, ...],
    buffer_idx: int,
) -> int:
  """Adds a tensor to the FlatBuffers."""
  name_offset = builder.CreateString(name)
  schema_fb.TensorStartShapeVector(builder, len(shape))
  for dim in reversed(shape):
    builder.PrependInt32(dim)
  shape_offset = builder.EndVector()
  schema_fb.TensorStart(builder)
  schema_fb.TensorAddName(builder, name_offset)
  schema_fb.TensorAddShape(builder, shape_offset)
  schema_fb.TensorAddType(builder, schema_fb.TensorType.FLOAT32)
  schema_fb.TensorAddBuffer(builder, buffer_idx)
  tensor_offset = schema_fb.TensorEnd(builder)
  return tensor_offset


def _lora_to_flatbuffers(lora: LoRA) -> bytearray:
  """Converts LoRA to FlatBuffers."""
  tensors, (names, _) = _flatten_lora(lora)
  # Need to manually add the "lora_" prefix to the names here. The export will
  # add the prefix automatically.
  names = [f"lora_{name}" for name in names]
  builder = flatbuffers.Builder(4096)

  # Convention to add an empty buffer in the beginning.
  buffer_offsets = [_add_buffer(builder, None)]
  for tensor in tensors:
    buffer_offsets.append(
        _add_buffer(builder, tensor.detach().type(torch.float32).numpy())
    )

  schema_fb.ModelStartBuffersVector(builder, len(buffer_offsets))
  for buffer_offset in reversed(buffer_offsets):
    builder.PrependUOffsetTRelative(buffer_offset)
  buffers_offset = builder.EndVector()

  tensor_offsets = []
  for i, (name, tensor) in enumerate(zip(names, tensors)):
    # Note that the zeroth buffer is empty and reserved for the convention.
    tensor_offsets.append(_add_tensor(builder, name, tensor.shape, i + 1))

  schema_fb.SubGraphStartTensorsVector(builder, len(tensor_offsets))
  for tensor_offset in reversed(tensor_offsets):
    builder.PrependUOffsetTRelative(tensor_offset)
  tensors_offset = builder.EndVector()

  string_offset = builder.CreateString("lora_params")
  schema_fb.SubGraphStart(builder)
  schema_fb.SubGraphAddName(builder, string_offset)
  schema_fb.SubGraphAddTensors(builder, tensors_offset)
  subgraph_offset = schema_fb.SubGraphEnd(builder)

  schema_fb.ModelStartSubgraphsVector(builder, 1)
  builder.PrependUOffsetTRelative(subgraph_offset)
  subgraphs_offset = builder.EndVector()

  # Add metadata for "lora_rank"
  metadata_name_offset = builder.CreateString("lora_rank")
  schema_fb.MetadataStart(builder)
  schema_fb.MetadataAddName(builder, metadata_name_offset)
  schema_fb.MetadataAddBuffer(builder, lora.get_rank())
  metadata_offset = schema_fb.MetadataEnd(builder)

  schema_fb.ModelStartMetadataVector(builder, 1)
  builder.PrependUOffsetTRelative(metadata_offset)
  metadata_vector_offset = builder.EndVector()

  string_offset = builder.CreateString("lora_params")
  schema_fb.ModelStart(builder)
  schema_fb.ModelAddVersion(builder, _TFLITE_SCHEMA_VERSION)
  schema_fb.ModelAddDescription(builder, string_offset)
  schema_fb.ModelAddBuffers(builder, buffers_offset)
  schema_fb.ModelAddSubgraphs(builder, subgraphs_offset)
  schema_fb.ModelAddMetadata(builder, metadata_vector_offset)
  model_offset = schema_fb.ModelEnd(builder)
  builder.Finish(model_offset, file_identifier=_TFLITE_FILE_IDENTIFIER)
  flatbuffer_model = builder.Output()

  return flatbuffer_model
