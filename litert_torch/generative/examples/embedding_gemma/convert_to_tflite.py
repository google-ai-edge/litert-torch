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
"""Example of converting EmbeddingGemma-300M model to TFLite."""

import os

from absl import app
import litert_torch as at
from litert_torch.generative.examples.embedding_gemma import embedding_gemma
from litert_torch.generative.utilities import converter as generative_converter
import torch

flags = generative_converter.define_conversion_flags(
    model_name="embedding_gemma"
)
flags.DEFINE_bool(
    "final_l2_norm",
    True,
    "Whether to apply final L2 normalization to the embeddings.",
)
FLAGS = flags.FLAGS


def main(_):
  model = embedding_gemma.build_model(
      FLAGS.checkpoint_path, final_l2_norm=FLAGS.final_l2_norm
  )
  model.eval()
  seq_len = max(FLAGS.prefill_seq_lens)

  sample_inputs = (
      torch.ones(1, seq_len, dtype=torch.long),  # tokens
      torch.ones(1, seq_len, dtype=torch.long),  # attention_mask
  )

  quant_config = generative_converter.get_quant_recipe_from_flag(
      FLAGS.quantize, model.config
  )
  edge_model = at.convert(
      model,
      sample_inputs,
      quant_config=quant_config,
  )

  output_dir = FLAGS.output_path
  quant_suffix = generative_converter.create_quantize_suffix(FLAGS.quantize)
  output_filename = f"{FLAGS.output_name_prefix}_{quant_suffix}.tflite"
  output_path = os.path.join(output_dir, output_filename)
  edge_model.export(output_path)
  print(f"TFLite model successfully saved to {output_path}")


if __name__ == "__main__":
  app.run(main)
