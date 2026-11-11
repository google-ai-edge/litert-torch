from absl import app
from ai_edge_torch.generative.utilities import converter
from ai_edge_torch.generative.utilities import export_config
from ai_edge_torch.generative.utilities import loader
import torch
from ai_edge_torch.generative.examples.smalvlm import smalvlm

flags = converter.define_conversion_flags('smalvlm-256m-instruct')

flags.DEFINE_bool(
    'do_image_splitting',
    True,
    'If true, model will expect image in pixel_values as [1, 13, 3, 512, 512].'
    ' Otherwise, as [1, 1, 3, 512, 512].',
)


def main(_):
  checkpoint_path = flags.FLAGS.checkpoint_path
  pytorch_model = smalvlm.build_model(
      checkpoint_path=checkpoint_path,
      custom_loader=loader.maybe_get_custom_loader(
          checkpoint_path, flags.FLAGS.custom_checkpoint_loader
      ),
      mask_cache_size=flags.FLAGS.kv_cache_max_len,
  )

  config = pytorch_model.image_encoder.config.image_embedding
  converter.convert_to_tflite(
      pytorch_model,
      output_path=flags.FLAGS.output_path,
      output_name_prefix=f'{flags.FLAGS.output_name_prefix}',
      prefill_seq_len=flags.FLAGS.prefill_seq_lens,
      kv_cache_max_len=flags.FLAGS.kv_cache_max_len,
      pixel_values_size=torch.Size([
          1,
          13 if flags.FLAGS.do_image_splitting else 1,
          config.channels,
          config.image_size,
          config.image_size,
      ]),
      pixel_seq_len=(config.image_size // config.patch_size) ** 2,
      quantize=flags.FLAGS.quantize,
      config=pytorch_model.config.decoder_config,
      export_config=export_config.get_from_flags(),
  )


if __name__ == '__main__':
  app.run(main)
