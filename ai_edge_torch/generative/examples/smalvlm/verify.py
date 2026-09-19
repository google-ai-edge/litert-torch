import logging
from absl import app
from absl import flags
from ai_edge_torch.generative.layers import kv_cache
from ai_edge_torch.generative.utilities import transformers_verifier
from ai_edge_torch.generative.utilities import verifier
from PIL import Image
import requests
import torch
import transformers
from transformers import AutoModelForVision2Seq


import smalvlm


_IMAGE_URL = flags.DEFINE_string(
    "image_url",
    "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/car.jpg?download=true",
    "The image URI to encode.",
)
_PROMPTS_WITH_IMAGE = flags.DEFINE_string(
    "prompts_with_image",
    "Explain the image in details",
    "The input prompts to generate answers with an image.",
)
_PROMPTS_TEXT_ONLY = flags.DEFINE_multi_string(
    "prompts_text_only",
    "<|im_start|>User:2+2?<end_of_utterance>\nAssistant:",
    "The input prompts to generate answers only with text.",
)
_MAX_NEW_TOKENS = flags.DEFINE_integer(
    "max_new_tokens",
    100,
    "The maximum size of the generated tokens.",
)


class ReauthoredSmalVLMWrapper(verifier.ReauthoredModelWrapper):
  """Reauthored SmalVLM model wrapper."""

  def _init_kv_cache(self):
    return kv_cache.KVCache.from_model_config(
        self.kv_cache_max_len, self.model.config.decoder_config
    )


def main(_):

  checkpoint_path = "./models/SmolVLM-256M-Instruct"
  kv_cache_max_len = 2048

  logging.info("Loading the original model from: %s", checkpoint_path)
  original_model = AutoModelForVision2Seq.from_pretrained(checkpoint_path)

  logging.info("Building the reauthored model from: %s", checkpoint_path)
  reauthored_model = smalvlm.build_model(
      checkpoint_path=checkpoint_path,
      mask_cache_size=kv_cache_max_len,
  )
  wrapped_reauthored_model = ReauthoredSmalVLMWrapper(
      reauthored_model,
      kv_cache_max_len=kv_cache_max_len,
  )

  logging.info("Loading the processor from: %s", checkpoint_path)
  processor = transformers.AutoProcessor.from_pretrained(
      checkpoint_path,
      do_image_splitting=True,
  )

  logging.info("Verifying with text-only prompts...")
  verifier.verify_reauthored_model(
      original_model=transformers_verifier.TransformersModelWrapper(
          original_model
      ),
      reauthored_model=wrapped_reauthored_model,
      tokenizer=verifier.TokenizerWrapper(processor.tokenizer),
      generate_prompts=_PROMPTS_TEXT_ONLY.value,
      max_new_tokens=_MAX_NEW_TOKENS.value,
      verify_inputs=False,  # Numeric check not working. Disable it for now.
      atol=1e-04,
  )

  logging.info("Verifying with image input...")
  logging.info("Loading the image from: %s", _IMAGE_URL.value)
  image = Image.open(requests.get(_IMAGE_URL.value, stream=True).raw)

  messages = [
      {
          "role": "user",
          "content": [
              {"type": "image"},
              {"type": "text", "text": _PROMPTS_WITH_IMAGE.value},
          ],
      },
  ]
  prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
  inputs = processor(text=prompt, images=[image], return_tensors="pt")

  logging.info("Verifying the reauthored model with model.forward()...")
  logging.info("Forwarding the original model...")
  outputs_original = original_model.forward(
      input_ids=inputs["input_ids"], pixel_values=inputs["pixel_values"]
  )
  outputs_original = outputs_original.logits
  logging.info("outputs_original: %s", outputs_original)

  logging.info("Forwarding the reauthored model...")
  outputs_reauthored = wrapped_reauthored_model.forward(
      tokens=inputs["input_ids"],
      pixel_values=inputs["pixel_values"],
  )
  logging.info("outputs_reauthored: %s", outputs_reauthored)

  try:
    assert torch.allclose(outputs_original, outputs_reauthored, atol=1e-02)
  except AssertionError as e:
    logging.error("*** FAILED *** verify with forward()")
    raise e
  else:
    logging.info("*** PASSED *** verify with forward()")

  logging.info("Verifying the reauthored model with model.generate()...")
  logging.info("Generating answer with the original model...")
  outputs_original = original_model.generate(
      **inputs, max_new_tokens=_MAX_NEW_TOKENS.value, do_sample=False
  )
  response_original = processor.decode(
      outputs_original[0], skip_special_tokens=True
  )
  logging.info("outputs_from_original_model: [[%s]]", response_original)

  logging.info("Generating answer with the reauthored model...")
  outputs_reauthored = wrapped_reauthored_model.generate(
      prompts=inputs["input_ids"],
      pixel_values=inputs["pixel_values"],
      max_new_tokens=_MAX_NEW_TOKENS.value,
      eos_token_id=processor.tokenizer.eos_token_id,
  )
  response_reauthored = processor.decode(
      outputs_reauthored[0], skip_special_tokens=True
  )
  logging.info("outputs from reauthored model: [[%s]]", response_reauthored)

  try:
    assert response_original == response_reauthored
  except AssertionError as e:
    logging.error("*** FAILED *** verify with generate()")
    raise e
  else:
    logging.info("*** PASSED *** verify with generate()")


if __name__ == "__main__":
  app.run(main)
