"""Utils for verifying the SmolVLM model."""

import logging
import os
import pathlib

from ai_edge_torch.generative.utilities import loader
from ai_edge_torch.generative.utilities import transformers_verifier
from ai_edge_torch.generative.utilities import verifier
import transformers


from text_model import build_model


DEFAULT_PROMPTS = ["What is the meaning of life?"]


def verify_smolvlm_text(
    checkpoint_dir: str,
    weight_filename: str = "model.safetensors",
    max_new_tokens: int = 30,
    initialize_from_local: bool = True,
    prompts: list[str] | None = None,
) -> bool:
  """Verifies the reauthored SmoLLM model with a custom loader."""
  logging.info("Loading the original model from: %s", checkpoint_dir)
  original_model = transformers.AutoModelForVision2Seq.from_pretrained(
      checkpoint_dir
  )

  logging.info("Building the reauthored model from: %s", checkpoint_dir)
  custom_loader = (
      None
      if initialize_from_local
      else loader.get_custom_loader("", "safetensors")
  )

  if initialize_from_local:
    # Locate the cached dir.
    cached_config_file = transformers.utils.cached_file(
        checkpoint_dir, transformers.utils.CONFIG_NAME
    )
    reauthored_checkpoint = pathlib.Path(cached_config_file).parent
  else:
    reauthored_checkpoint = os.path.join(checkpoint_dir, weight_filename)

  logging.info("Building the reauthored model from: %s", reauthored_checkpoint)
  reauthored_model = build_model(
      checkpoint_path=reauthored_checkpoint,
      custom_loader=custom_loader,
      mask_cache_size=1024,
  )

  logging.info("Loading the tokenizer from: %s", checkpoint_dir)
  tokenizer = transformers.AutoTokenizer.from_pretrained(checkpoint_dir)
  return verifier.verify_reauthored_model(
      original_model=transformers_verifier.TransformersModelWrapper(
          original_model
      ),
      reauthored_model=verifier.ReauthoredModelWrapper(reauthored_model),
      tokenizer=verifier.TokenizerWrapper(tokenizer),
      generate_prompts=DEFAULT_PROMPTS if prompts is None else prompts,
      max_new_tokens=max_new_tokens,
      atol=1e-04,
  )


from absl import app
from absl import flags

_PROMPTS = flags.DEFINE_multi_string(
    "prompts",
    "What is the meaning of life?",
    "The input prompts to generate answers.",
)
_MAX_NEW_TOKENS = flags.DEFINE_integer(
    "max_new_tokens",
    30,
    "The maximum size of the generated tokens.",
)


def main(_):

  verify_smolvlm_text(
      checkpoint_dir="./models/SmolVLM-256M-Instruct",
      max_new_tokens=_MAX_NEW_TOKENS.value,
      prompts=_PROMPTS.value,
  )


if __name__ == "__main__":
  app.run(main)
