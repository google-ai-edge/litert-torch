"""Verifies the reauthored image encoder of SmolVLM-256M model."""

import logging
from absl import app
from absl import flags
from PIL import Image
import requests
import torch
import transformers
from transformers import AutoModelForVision2Seq


from image_encoder import build_image_encoder


_IMAGE_URL = flags.DEFINE_string(
    "image_url",
    "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/car.jpg?download=true",
    "The image URI to encode.",
)


_CHECKPOINT = "./models/SmolVLM-256M-Instruct"


def main(_):

  original_full_model = AutoModelForVision2Seq.from_pretrained(
      _CHECKPOINT,
  )

  original_vision_model = original_full_model.model.vision_model

  logging.info("Building the reauthored model from: %s", _CHECKPOINT)
  reauthored_model = build_image_encoder(_CHECKPOINT)

  logging.info("Loading the processor from: %s", _CHECKPOINT)
  # It works only when GemmaTokenizerFast is available. In some environments,
  # use_fast=False doeesn't work either if the tokenizer cannot load the
  # sentencepiece model file properly.
  processor = transformers.AutoProcessor.from_pretrained(
      _CHECKPOINT,
      do_image_splitting=False,
  )

  logging.info("Loading the image from: %s", _IMAGE_URL.value)
  image = Image.open(requests.get(_IMAGE_URL.value, stream=True).raw)
  pixel_values = processor(images=image, return_tensors="pt")["pixel_values"]
  pixel_values = pixel_values.squeeze(dim=0)

  logging.info("Forwarding the original model...")
  outputs_original = original_vision_model.forward(pixel_values=pixel_values)
  outputs_original = outputs_original.last_hidden_state
  logging.info("outputs_original: %s", outputs_original)

  logging.info("Forwarding the reauthored model...")
  outputs_reauthored = reauthored_model.forward(pixel_values=pixel_values)
  logging.info("outputs_reauthored: %s", outputs_reauthored)

  try:
    assert torch.allclose(
        outputs_original, outputs_reauthored, atol=1e-03, rtol=1e-04
    )
  except AssertionError as e:
    logging.error("*** FAILED *** verify with an image")
    raise e
  else:
    logging.info("*** PASSED *** verify with an image")


if __name__ == "__main__":
  app.run(main)
