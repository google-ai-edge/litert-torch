
<!--ts-->
* [API Walkthrough](#api-walkthrough)
   * [Conversion](#conversion)
   * [Inference](#inference)
   * [Serialization](#serialization)
   * [Importing a model](#importing-a-model)
   * [Multi-Signature Conversion](#multi-signature-conversion)
   * [Quantization](#quantization)
   * [Providing a Wrapper](#providing-a-wrapper)
   * [Convert Model with NHWC (Channel Last) Inputs/Outputs](#convert-model-with-nhwc-channel-last-inputsoutputs)
* [Debugging &amp; Reporting Errors](#debugging--reporting-errors)
   * [Error during torch.export.export](#error-during-torchexportexport)
   * [Error during ExportedProgram to edge model lowering](#error-during-exportedprogram-to-edge-model-lowering)
* [Visualization](#visualization)
* [Use Torch XLA Conversion Backend (Legacy)](#use-torch-xla-conversion-backend-legacy)
   * [Update LD_LIBRARY_PATH if necessary](#update-ld_library_path-if-necessary)

<!-- Created by https://github.com/ekalinin/github-markdown-toc -->

<!--te-->

# API Walkthrough

This section walks through the end-to-end process of preparing a PyTorch model for on-device deployment.

We'll use the `resnet18` model from the PyTorch [torchvision](https://pytorch.org/vision/stable/index.html) package as an example. This model can be executed in PyTorch as below:

```python
import torch
import torchvision
resnet18 = torchvision.models.resnet18(torchvision.models.ResNet18_Weights.IMAGENET1K_V1).eval()
sample_inputs = (torch.randn(1, 3, 224, 224),)
torch_output = resnet18(*sample_inputs)
```

## Conversion
`litert_torch.convert()` converts a PyTorch model to an on-device (Edge) model.
The conversion process also requires sample inputs for tracing and shape
inference, passed in as a tuple. As an example, if the PyTorch model receives 3
tensors as positional arguments, the `convert` function receives 1 tuple with 3
entries.

- **Note 1:** The source PyTorch model needs to be compliant with
[`torch.export`](https://pytorch.org/docs/stable/export.html) introduced in
PyTorch 2.1.0 .

- **Note 2:** `convert` expects a `torch.nn.Module` with a `forward` function
that receives tensors as arguments and returns
tensors as outputs. If your model has a different interface, you need to provide a model wrapper, as demonstrated in the [Providing a Wrapper](#providing-a-wrapper) section.

- **Note 3:** `convert` does not support passing keyword arguments to the model.

```python
import litert_torch

# Note that we are setting the model to evaluation mode prior to conversion.
edge_model = litert_torch.convert(resnet18.eval(), sample_inputs)
```

## Inference

Once the model is converted, it is ready for inference with the
[TFLite runtime](https://www.tensorflow.org/lite/guide/inference). Prior to
deployment on-device, the outputs from PyTorch and the edge model can be
compared in Python as a smoke check for the converted model.

```python
import numpy as np

edge_output = edge_model(*sample_inputs)
assert np.allclose(torch_output.detach().numpy(), edge_output, atol=1e-5)
```

## Serialization
The on-device prepared model provides an `export` function which can be used to
serialize the model as a [TFLite](https://www.tensorflow.org/lite/guide)
Flatbuffers file (`.tflite`) which can be used
[for deployment](https://www.tensorflow.org/lite/guide/inference).

```python
edge_model.export('resnet.tflite')
```

## Importing a model
A model serialized via `export` or any TFLite Flatbuffers file can be imported
into `litert_torch` as follows:

```python
imported_edge_model = litert_torch.load('resnet.tflite')

# Once imported, you can run the model with an input.
imported_edge_model(*sample_inputs)
```

## Multi-Signature Conversion

Sometimes, it is desirable to have multiple PyTorch modules converted into one
edge model. This is often the case when a model comprises multiple components
that share weights.

[Signatures](https://www.tensorflow.org/lite/guide/signatures) are a TF Lite
feature to address this.

The API for multi-signature conversion with `litert_torch` is as follows:
```python
inputs_1 = (...,)
inputs_2 = (...,)

edge_model = litert_torch
            .signature("input1", model, inputs_1)
            .signature("input2", model, inputs_2)
            .convert()

# Run each signature separately by providing the signature_name as a keyword argument.
edge_model(*inputs_1, signature_name="input1")
edge_model(*inputs_2, signature_name="input2")
```

## Quantization

Following is the code snippet to quantize a model with [PT2E
quantization](https://docs.pytorch.org/ao/stable/tutorials_source/pt2e_quant_ptq.html)
using the `litert_torch` backend.

```python
from torchao.quantization.pt2e.quantize_pt2e import prepare_pt2e, convert_pt2e
from litert_torch.quantize.pt2e_quantizer import get_symmetric_quantization_config
from litert_torch.quantize.pt2e_quantizer import PT2EQuantizer
from litert_torch.quantize.quant_config import QuantConfig

pt2e_quantizer = PT2EQuantizer().set_global(
    get_symmetric_quantization_config(is_per_channel=True, is_dynamic=True)
)

# > For pytorch 2.6+
pt2e_torch_model = torch.export.export(torch_model, sample_args).module()
# > For pytorch 2.5 and before
# from torch._export import capture_pre_autograd_graph
# pt2e_torch_model = capture_pre_autograd_graph(torch_model, sample_args)

pt2e_torch_model = prepare_pt2e(pt2e_torch_model, pt2e_quantizer)

# Run the prepared model with sample input data to ensure that internal observers are populated with correct values
pt2e_torch_model(*sample_args)

# Convert the prepared model to a quantized model
pt2e_torch_model = convert_pt2e(pt2e_torch_model, fold_quantize=False)

# Convert to an litert_torch model
pt2e_drq_model = litert_torch.convert(pt2e_torch_model, sample_args, quant_config=QuantConfig(pt2e_quantizer=pt2e_quantizer))
```

To quantize a converted model, use [LiteRT
Quantizer](https://github.com/google-ai-edge/LiteRT/tree/main/litert_quantizer):

```python
import litert_torch
from litert_quantizer import quantizer, recipe

# Step 1: Convert the PyTorch model to a float32 LiteRT model.
edge_model = litert_torch.convert(torch_model, sample_args)
edge_model.export("model_float32.tflite")

# Step 2: Quantize the float32 model with LiteRT Quantizer.
qt = quantizer.Quantizer("model_float32.tflite")
qt.load_quantization_recipe(recipe.dynamic_wi8_afp32())
qt.quantize().export_model("model_quantized.tflite")
```

Post-training quantization through TFLite converter flags
(`_ai_edge_converter_flags`) is deprecated. See the [migration
guide](https://developers.google.com/edge/litert/quantization/tflq_to_litertq_migration).

## Providing a Wrapper

`litert_torch.convert` expects an `nn.Module` with a `forward` function that
receives tensors as positional arguments and returns a tensor, or multiple
tensors in a Python list or tuple. If you have a model with a different
interface, you will need to provide a wrapper.

As an example, let's say `MyModel` receives only `kwargs` and returns a custom
object. Here is how the mentioned wrapper would look:

```python
class MyModelWrapper(torch.nn.Module):
  def __init__(self):
    super().__init__()
    self.m = MyModel()

  def forward(self, tensor1, tensor2):
    custom_output_object = self.m(arg1=tensor1, arg2=tensor2)
    return custom_output_object.out_tensor1, custom_output_object.out_tensor2
```

The instance in evaluation mode, `MyModelWrapper().eval()`, would be the right argument to pass to `litert_torch.convert`.

## Convert Model with NHWC (Channel Last) Inputs/Outputs

`litert_torch.to_channel_last_io` is a helper function facilitates the conversion of
PyTorch models (typically using NCHW channel first ordering) to TFLite models with
channel last (NHWC) input/output layouts. It achieves this by wrapping the original model
with layout transformation transposes, ensuring compatibility with target
deployment environments. This is particularly useful for deploying models,
such as image classifiers, to mobile environments that expect NHWC (channel last)
image data.

Here is an example of converting ResNet18 with NHWC image input:
```python
import torch
import torchvision
import litert_torch

# Use resnet18 with pre-trained weights.
resnet18 = torchvision.models.resnet18(torchvision.models.ResNet18_Weights.IMAGENET1K_V1)

# Transform the first input to NHWC.
nhwc_resnet18 = litert_torch.to_channel_last_io(resnet18, args=[0])

# Convert the transformed model with NHWC input(s).
edge_model = litert_torch.convert(nhwc_resnet18, (torch.randn(1, 224, 224, 3),))
edge_model.export("resnet18.tflite")
```

More examples of usage can be found [here](https://github.com/google-ai-edge/litert-torch/blob/main/litert_torch/_convert/test/test_to_channel_last_io.py).

# Debugging & Reporting Errors

Failure of `litert_torch.convert(...)` can happen in a multiple different steps
with verbose and potentially hard to understand error messages.

The two high-level steps that users should be aware of are
 1. [torch.export](https://pytorch.org/docs/stable/export.html) to convert
    PyTorch model to an [ExportedProgram](https://pytorch.org/docs/stable/export.html#torch.export.ExportedProgram)

 1. Lowering from ExportedProgram to an [edge\_model](https://github.com/google-ai-edge/litert-torch/blob/main/litert_torch/model.py).

In case of a `convert` failure, please use our `find_culprits` tool to help
narrow down the issue and generate a minimal PyTorch program that reproduces the
failure (in some cases).

`find_culprits` can be given the same parameters as `convert`:

```python
from litert_torch.debug import find_culprits

culprits = find_culprits(model.eval(), args)
culprit = next(culprits)
culprit.print_code()

```

## Error during torch.export.export

In this case `print_code()` will provide all the logs from `torch.export.export`
followed by an error message confirming the error type.
```
ValueError: Your model is not exportable by torch.export.export. Please modify your model to be torch-exportable first.
```

The fix for these errors involves changing the model source to be compliant
with `torch.export` and is not a bug in `litert_torch.convert`. Please refer
to [PyTorch torch.export doc](https://pytorch.org/docs/stable/export.html)
for more information.

## Error during ExportedProgram to edge model lowering

For errors after we have an ExportedProgram, `find_culprits` can provide
a minimal reproduction code sample that can be attached to a GitHub issue.

Below is a code snippet that causes such a failure.

```python
import torch
import torchaudio
import litert_torch

model = torchaudio.models.ConvTasNet()
args = (torch.rand((1, 1, 256)),)
litert_torch.convert(model.eval(), args)
```

To debug the error, call `litert_torch.debug.find_culprits` with the same arguments
provided to `litert_torch.convert(...)` to get a generator of culprits.

```python
from litert_torch.debug import find_culprits

culprits = find_culprits(model, args)
```

Next, print a Python code snippet that reproduces the error with.

```python
culprit = next(culprits)
culprit.print_code()
```

Which prints the following to the console.

```python
import torch
from torch import device
import litert_torch

class CulpritGraphModule(torch.nn.Module):
    def forward(self, arg0_1: "f32[512, 1, 16]", arg1_1: "f32[2, 512, 33]"):
        # File: /opt/venv/lib/python3.10/site-packages/torchaudio/models/conv_tasnet.py:300 in forward, code: decoded = self.decoder(masked)  # B*S, 1, L'
        convolution: "f32[2, 1, 256]" = torch.ops.aten.convolution.default(arg1_1, arg0_1, None, [8], [8], [1], True, [0], 1);  arg1_1 = arg0_1 = None
        return (convolution,)

_args = (
    torch.randn((512, 1, 16,), dtype=torch.float32),
    torch.randn((2, 512, 33,), dtype=torch.float32),
)

_edge_model = litert_torch.convert(CulpritGraphModule().eval(), _args) # conversion should fail
```

You can attach the code snippet to a GitHub issue, after:

- Confirming that the generated code snippet fails conversion with the same error as the original program.
- Removing any sensitive information before reporting the issue with the code snippets to us.
- Note that the culprit finder tool overwrites weights and inputs with random values in the generated code.

You can also find and print all culprits at once:

```python
for culprit in find_culprits(model, args):
  culprit.print_code()
```

# Visualization
Once the exported TFLite model is obtained, you can visualize the model structure with [Model Explorer](https://github.com/google-ai-edge/model-explorer).

```
pip install ai-edge-model-explorer
model-explorer 'resnet.tflite'
```

# Use Torch XLA Conversion Backend (Legacy)

LiteRT Torch has been switched to a modern conversion backend for better
on-device compatibility and performance. However, if you encounter compatibility
issues, you can optionally switch to the legacy Torch XLA backend:

```bash
# Install litert-torch with torch-xla dependency
pip install --pre litert-torch-nightly[torch-xla]

# Enable torch-xla as the LiteRT Torch backend
export USE_TORCH_XLA=1
```

## Update LD_LIBRARY_PATH if necessary

Torch XLA builds a shared library, `_XLAC.so` that needs to link to the version of Python
it was built with (currently 3.10 or 3.11). In order to ensure that `import _XLAC` can succeed,
update the LD_LIBRARY_PATH to the lib directory of your Python environment:

```bash
export LD_LIBRARY_PATH=<path to Python installation>/lib:$LD_LIBRARY_PATH
```
