# Copyright 2026 The LiteRT Torch Authors.
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
"""HuggingFace `transformers.exporters` backend powered by `export_hf`.

Registers `"litert"` with `transformers.exporters.AutoExportConfig` and
`transformers.exporters.AutoHfExporter`:

  # Registers the "litert" exporter backend.
  from litert_torch.generative.export_hf import transformers_exporter
  from transformers.exporters import AutoExportConfig, AutoHfExporter

  cfg = AutoExportConfig.from_dict({"export_format": "litert"})
  exporter = AutoHfExporter.from_config(cfg)
  artifacts = exporter.export_for_generation(model, config=cfg, tokenizer=tok)
"""

from __future__ import annotations

from collections.abc import Mapping, MutableMapping
import contextlib
import dataclasses
import os
import tempfile
from typing import Any, cast

from litert_torch import model as lrt_model_lib
from litert_torch._convert import interface as converter_utils
from litert_torch.generative.export_hf import export
# Import for side effects: register legacy patches and ModelExtensions.
from litert_torch.generative.export_hf import model_ext as _
from litert_torch.generative.export_hf.core import export_lib
from litert_torch.generative.export_hf.core import exportable_module_config
from litert_torch.generative.export_hf.model_ext import builtin_extensions as _
from litert_torch.generative.export_hf.model_ext import patches as model_ext_patches
import torch
from transformers.exporters import auto as exporters_auto
from transformers.exporters import base as exporters_base
from transformers.exporters import configs as exporters_configs
from transformers.exporters import exporter_dynamo
from transformers.exporters import utils as exporters_utils


def _resolve_model_type(model: torch.nn.Module) -> str:
  """Returns the text or top-level `model_type` for `model`, or `""`."""
  model_config = getattr(model, "config", None)
  if model_config is None:
    return ""
  text_config = getattr(model_config, "text_config", None)
  if text_config is not None:
    text_type = getattr(text_config, "model_type", None)
    if text_type:
      return str(text_type)
  return str(getattr(model_config, "model_type", "") or "")


@dataclasses.dataclass
class LiteRTConfig(exporters_configs.ExportConfigMixin):
  """Thin `ExportConfigMixin` delegating generation options to `export_hf`.

  Any `ExportableModuleConfig` field (such as `prefill_lengths`,
  `cache_length`, `quantization_recipe`, `aot_backend`, `aot_soc_model`,
  `calibration_dataset_dir`, `bundle_litert_lm`, `delete_in_memory_params`)
  may be passed either inside `export_kwargs` or as a top-level key in
  `LiteRTConfig.from_dict(...)` / `AutoExportConfig.from_dict(...)`.

  Attributes:
    export_format: Registered exporter name (`"litert"`).
    target: Hardware target preset validated by `export.parse_target`:
      `"cpu_gpu"` (default; `"cpu"` and `"gpu"` are accepted as aliases),
      `"npu"`, or `"npu:<vendor>"`.
    task: Export task (`"auto"` or an `ExportTask` value).
    output_dir: Optional destination directory for exported artifacts. When
      `None`, a temporary directory is created.
    dynamic: Whether to trace with dynamic shapes in `export()`.
    strict: Whether to run `torch.export` in strict mode.
    use_v2: Whether to use LiteRT Converter V2 (defaults to `True`).
    quantizer: Optional upstream `ExportQuantizer` (reserved for `HfExporter`
      interface compatibility).
    export_kwargs: Additional `ExportableModuleConfig` / `export.export`
      overrides forwarded verbatim to `export.export()`.
  """

  export_format: Any = "litert"
  target: str = export.TARGET_CPU_GPU
  task: str = export.TASK_AUTO
  output_dir: str | None = None
  dynamic: bool = False
  strict: bool = False
  use_v2: bool = True
  quantizer: Any = None
  export_kwargs: dict[str, Any] = dataclasses.field(default_factory=dict)

  def __post_init__(self) -> None:
    self.target = export.parse_target(self.target).canonical

  @classmethod
  def from_dict(cls, config_dict: Mapping[str, Any]) -> LiteRTConfig:
    """Instantiates `LiteRTConfig`, routing `export_hf` keys to `export_kwargs`."""
    data = dict(config_dict)
    if "backend" in data and "target" not in data:
      data["target"] = data.pop("backend")
    own_fields = {f.name for f in dataclasses.fields(cls)}
    init_kwargs: dict[str, Any] = {}
    extra_export_kwargs: dict[str, Any] = dict(data.pop("export_kwargs", {}))
    for key, value in data.items():
      if key in own_fields:
        init_kwargs[key] = value
      else:
        extra_export_kwargs[key] = value
    if extra_export_kwargs:
      init_kwargs["export_kwargs"] = extra_export_kwargs
    return cls(**init_kwargs)


@contextlib.contextmanager
def _prepare_model_for_litert(
    model: torch.nn.Module,
    target_family: str,
    em_config: exportable_module_config.ExportableModuleConfig,
):
  """Enters exporter patches and `ModelExtension` hooks around tracing."""
  model_type = _resolve_model_type(model)
  with contextlib.ExitStack() as stack:
    stack.enter_context(exporters_utils.apply_patches("litert"))
    stack.enter_context(
        exporters_utils.apply_patches(f"litert.{target_family}")
    )
    stack.enter_context(export_lib.patch_builtin_tuple_for_export())
    if model_type:
      stack.enter_context(
          model_ext_patches.get_patch_context(model_type, em_config)
      )
      stack.enter_context(
          model_ext_patches.patch_model(model, model_type, em_config)
      )
    yield model


def _coerce_config(
    config: (
        LiteRTConfig
        | exporters_configs.ExportConfigMixin
        | Mapping[str, Any]
        | None
    ),
) -> LiteRTConfig:
  """Normalizes `None`, a mapping, or a `LiteRTConfig` to `LiteRTConfig`."""
  if config is None:
    return LiteRTConfig()
  if isinstance(config, LiteRTConfig):
    return config
  if isinstance(config, Mapping):
    return LiteRTConfig.from_dict(config)
  raise TypeError(
      f"Expected config to be a LiteRTConfig or mapping, got {type(config)}."
  )


class _FlatTensorWrapper(torch.nn.Module):
  """Wraps an `ExportedProgram` module so the converter only sees Tensors.

  Flattens pytree kwargs (such as `StaticCache` or dicts with `None` mask
  entries) and binds non-tensor leaves as constants so every converter input
  placeholder has tensor metadata.
  """

  def __init__(
      self,
      module: torch.nn.Module,
      sample_kwargs: Mapping[str, Any],
  ) -> None:
    super().__init__()
    self.module = module
    flat_values, self._tree_spec = torch.utils._pytree.tree_flatten(
        dict(sample_kwargs)
    )
    self._tensor_positions: list[int] = [
        idx
        for idx, value in enumerate(flat_values)
        if isinstance(value, torch.Tensor)
    ]
    self._const_values: list[Any] = list(flat_values)

  def tensor_inputs(self) -> tuple[torch.Tensor, ...]:
    """Returns the ordered tuple of leaf tensor sample inputs."""
    return tuple(self._const_values[i] for i in self._tensor_positions)

  def forward(self, *tensors: torch.Tensor) -> Any:
    leaves = list(self._const_values)
    for pos, tensor in zip(self._tensor_positions, tensors, strict=True):
      leaves[pos] = tensor
    kwargs = torch.utils._pytree.tree_unflatten(  # pylint: disable=protected-access
        leaves, self._tree_spec
    )
    return self.module(**kwargs)


class LiteRTExporter(exporters_base.HfExporter):
  """Exports HuggingFace models to LiteRT (`.tflite`) and `.litertlm`."""

  export_format = "litert"
  config_class = LiteRTConfig
  program_suffix = ".tflite"

  required_packages = ["torch"]
  min_versions = {"torch": "2.6.0"}

  def _lower_exported_program_to_litert(
      self,
      exported_program: torch.export.ExportedProgram,
      sample_inputs: Mapping[str, Any],
      config: LiteRTConfig,
      signature_name: str = "forward",
      output_path: str | None = None,
  ) -> lrt_model_lib.LiteRTModel:
    """Lowers a single `torch.export.ExportedProgram` to a `LiteRTModel`."""
    module = exported_program.module()
    for param in module.parameters():
      param.requires_grad_(False)
    wrapper = _FlatTensorWrapper(module, sample_inputs)

    converter = converter_utils.Converter()
    converter.add_signature(
        signature_name,
        wrapper,
        sample_args=wrapper.tensor_inputs(),
    )
    v2_kwargs: dict[str, Any] = {}
    if config.use_v2 and output_path is not None:
      v2_kwargs = {
          "use_v2": True,
          "output_file_path": output_path,
          "allow_reuse_intermediates": False,
          "_litert_converter_flags": {
              "fold_fp16_casts_into_fully_connected": True,
          },
      }
      if config.export_kwargs.get("keep_temporary_files"):
        v2_kwargs["export_dir"] = (
            os.path.splitext(output_path)[0] + "_intermediates"
        )
    with torch.no_grad(), export_lib.patch_builtin_tuple_for_export():
      return cast(
          lrt_model_lib.LiteRTModel,
          converter.convert(strict_export=config.strict, **v2_kwargs),
      )

  def export_program(
      self,
      model: Any,
      sample_inputs: MutableMapping[str, Any],
      config: (
          exporters_configs.ExportConfigMixin | Mapping[str, Any] | None
      ) = None,
      *,
      signature_name: str = "forward",
      output_path: str | None = None,
  ) -> tuple[lrt_model_lib.LiteRTModel, Any]:
    """Traces and lowers one forward graph to `(LiteRTModel, ExportMetadata)`."""
    litert_config = _coerce_config(config)
    work_dir = litert_config.output_dir or tempfile.mkdtemp(
        prefix="litert_export_"
    )
    os.makedirs(work_dir, exist_ok=True)
    target_family = export.parse_target(litert_config.target).family
    em_config = exportable_module_config.ExportableModuleConfig(
        model=getattr(model, "name_or_path", "") or "",
        output_dir=work_dir,
        work_dir=work_dir,
        use_rope_composite=False,
        use_v2=litert_config.use_v2,
    )

    with _prepare_model_for_litert(
        model, target_family, em_config
    ) as prepared_model:
      dynamo_exporter = exporter_dynamo.DynamoExporter()
      dynamo_config = exporters_configs.DynamoConfig(
          dynamic=litert_config.dynamic,
          strict=litert_config.strict,
      )
      export_program_fn = getattr(dynamo_exporter, "export_program", None)
      if export_program_fn is not None:
        exported_program, metadata = export_program_fn(
            cast(Any, prepared_model),
            sample_inputs,
            config=dynamo_config,
        )
      else:
        exported_program = dynamo_exporter.export(
            cast(Any, prepared_model),
            sample_inputs,
            config=dynamo_config,
        )
        metadata = None
      target_path = output_path
      if target_path is None and litert_config.use_v2:
        target_path = os.path.join(work_dir, f"{signature_name}.tflite")
      litert_model = self._lower_exported_program_to_litert(
          exported_program,
          sample_inputs,
          litert_config,
          signature_name=signature_name,
          output_path=target_path,
      )
      return litert_model, metadata

  @classmethod
  def save_program(
      cls, program: lrt_model_lib.LiteRTModel, path: str | os.PathLike[str]
  ) -> None:
    """Writes a `LiteRTModel` flatbuffer to `path`."""
    program.export(os.fspath(path))

  @classmethod
  def load_program(
      cls, path: str | os.PathLike[str]
  ) -> lrt_model_lib.LiteRTModel:
    """Loads a `LiteRTModel` flatbuffer from `path`."""
    return lrt_model_lib.LiteRTModel.load(os.fspath(path))

  def export(
      self,
      model: Any,
      sample_inputs: MutableMapping[str, Any],
      config: (
          exporters_configs.ExportConfigMixin | Mapping[str, Any] | None
      ) = None,
      *,
      signature_name: str = "forward",
      output_path: str | None = None,
  ) -> lrt_model_lib.LiteRTModel:
    """Exports a single `(model, sample_inputs)` forward graph to LiteRT.

    Supports non-LLM models (vision, audio, text encoders) as well as
    individual components from `decompose_for_generation` (`StaticCache`).

    Args:
      model: HuggingFace `PreTrainedModel` or `torch.nn.Module` to export.
      sample_inputs: Forward kwargs matching `model(**sample_inputs)`.
      config: `LiteRTConfig`, config mapping, or `None` for defaults.
      signature_name: Signature name inside the resulting LiteRT flatbuffer.
      output_path: Optional destination `.tflite` path when `use_v2=True`. When
        `None` and `use_v2=True`, writes `<output_dir>/<signature_name>.tflite`.

    Returns:
      The converted `LiteRTModel`.
    """
    litert_model, _ = self.export_program(
        model,
        sample_inputs,
        config=config,
        signature_name=signature_name,
        output_path=output_path,
    )
    return litert_model

  def export_for_generation(
      self,
      model: Any,
      sample_inputs: MutableMapping[str, Any] | None = None,
      config: (
          exporters_configs.ExportConfigMixin | Mapping[str, Any] | None
      ) = None,
      generation_config: Any = None,
      multi_token_decode: bool = False,
      tokenizer: Any = None,
  ) -> dict[str, Any]:
    """Exports a generative model via the shared `export_hf` pipeline.

    Delegates plan resolution, `ModelExtension.get_export_defaults`, task
    ordering, `.litertlm` packaging, and NPU calibration / static quantization /
    AOT compilation directly to `export.export()`.

    Args:
      model: Generative HuggingFace `PreTrainedModel` to export.
      sample_inputs: Accepted for `HfExporter` interface compatibility.
      config: `LiteRTConfig`, config mapping, or `None` for defaults.
      generation_config: Accepted for `HfExporter` interface compatibility.
      multi_token_decode: Accepted for `HfExporter` interface compatibility.
      tokenizer: Optional HuggingFace tokenizer. Passing a tokenizer enables
        `.litertlm` bundling automatically unless `bundle_litert_lm=False`.

    Returns:
      `{"litert_lm": <path>}` when bundled, or a dict of non-empty `.tflite`
      artifact paths when `bundle_litert_lm=False`.

    Raises:
      ValueError: If `bundle_litert_lm=True` without a `tokenizer`.
    """
    del sample_inputs, generation_config, multi_token_decode
    litert_config = _coerce_config(config)
    overrides = dict(litert_config.export_kwargs)
    overrides.setdefault("use_v2", litert_config.use_v2)

    if overrides.get("bundle_litert_lm") and tokenizer is None:
      raise ValueError(
          "`tokenizer` must be provided when `bundle_litert_lm=True`."
      )
    should_bundle = overrides.setdefault(
        "bundle_litert_lm", tokenizer is not None
    )

    output_dir = litert_config.output_dir or tempfile.mkdtemp(
        prefix="litert_lm_export_"
    )
    target_family = export.parse_target(litert_config.target).family

    with (
        exporters_utils.apply_patches("litert"),
        exporters_utils.apply_patches(f"litert.{target_family}"),
    ):
      plan = export.export(
          model=model,
          output_dir=output_dir,
          task=litert_config.task,
          target=litert_config.target,
          tokenizer=tokenizer,
          **overrides,
      )

    exported = plan.artifacts
    if exported is None:
      return {}
    if should_bundle:
      return {"litert_lm": exported.litert_lm_model_path}

    artifact_paths = {
        "prefill_decode": exported.prefill_decode_model_path,
        "embedder": exported.embedder_model_path,
        "auxiliary": exported.auxiliary_model_path,
        "vision_encoder": exported.vision_encoder_model_path,
        "vision_adapter": exported.vision_adapter_model_path,
        "audio_encoder": exported.audio_encoder_model_path,
        "audio_adapter": exported.audio_adapter_model_path,
    }
    return {k: v for k, v in artifact_paths.items() if v is not None}


def _register_litert_backend() -> None:
  """Registers LiteRT with `transformers.exporters` across v5.22+ and v5.24+."""
  register_backend = getattr(exporters_auto, "register_backend", None)
  if register_backend is not None:
    register_backend(LiteRTExporter)
  else:
    getattr(exporters_auto, "register_export_config")("litert")(LiteRTConfig)
    getattr(exporters_auto, "register_exporter")("litert")(LiteRTExporter)


_register_litert_backend()
