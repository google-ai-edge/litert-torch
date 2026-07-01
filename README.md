# LiteRT Torch

LiteRT Torch is a python library that supports converting PyTorch models into a
.tflite format, which can then be run with [LiteRT](https://ai.google.dev/edge/litert).
This enables applications for Android, iOS and IOT that can run models
completely on-device. LiteRT Torch offers broad CPU coverage, with initial GPU
and NPU support.  LiteRT Torch seeks to closely integrate with PyTorch,
building on top of torch.export() and providing good coverage of Core ATen
operators.

To get started converting PyTorch models to LiteRT, see additional details in
the [PyTorch converter](#pytorch-converter) section. For the particular case of
Large Language Models (LLMs) and transformer-based models, the [Generative
API](#generative-api) supports model authoring and quantization to enable
improved on device performance.

Although part of the same PyPi package, the PyTorch converter is a Beta release,
while the Generative API is an Alpha release. Please see the [release
notes](https://github.com/google-ai-edge/litert-torch/releases/) for additional
information.

## Installation

### Requirements and Dependencies

 * Python versions: `>=3.10` and `<3.14` (**Python 3.11 is highly recommended.** *Note: Python 3.14 currently has compatibility issues with typing in `torchao`.*)
 * Operating system: Linux
 * PyTorch: [![torch](https://img.shields.io/badge/torch->=2.4.0-blue)](https://pypi.org/project/torch/)
 * TensorFlow: [![tf-nightly](https://img.shields.io/badge/tf--nightly-latest-blue)](https://pypi.org/project/tf-nightly/)

<!-- requirement badges are updated by ci/update_nightly_versions.py -->

### Python Virtual Env

Set up a Python virtualenv (we strongly recommend Python 3.11):
```bash
python3.11 -m venv --prompt litert-torch venv
source venv/bin/activate
```

The latest stable release can be installed with (we include `torchvision` here to run the quickstart example below, and `ai-edge-litert` for CLI benchmarking tools):
```bash
pip install litert-torch torchvision ai-edge-litert
```

Alternately, the nightly version can be installed with:
```bash
pip install --pre litert-torch-nightly torchvision ai-edge-litert litert-cli-nightly
```

* The list of versioned releases can be seen [here](https://github.com/google-ai-edge/litert-torch/releases).
* The full list of PyPi releases (including nightly builds) can be seen [here](https://pypi.org/project/litert-torch/#history).

## PyTorch Converter
Here are the steps needed to convert a PyTorch model to a .tflite flatbuffer:

```python
import torch
import torchvision
import litert_torch

# Use resnet18 with pre-trained weights.
resnet18 = torchvision.models.resnet18(torchvision.models.ResNet18_Weights.IMAGENET1K_V1)

with torch.no_grad():
    sample_inputs = (torch.randn(1, 3, 224, 224),)

    # Convert and serialize PyTorch model to a .tflite flatbuffer. Note that we
    # are setting the model to evaluation mode prior to conversion.
    edge_model = litert_torch.convert(resnet18.eval(), sample_inputs)

edge_model.export("resnet18.tflite")
```

### Next Steps: Running the Model
Once exported, you can run this model on-device using the **LiteRT compiled model API** for native C++ or Java deployments, ensuring maximum performance across CPU, GPU, and NPU hardware accelerators.

The [getting started](docs/pytorch_converter/getting_started.ipynb) Jupyter
notebook gives an initial walkthrough of the conversion process and can be tried
out with Google Colab.

Additional technical details of the PyTorch Converter are [here](docs/pytorch_converter/README.md).

## Generative API
The LiteRT Torch Generative API is a Torch native library for authoring
mobile-optimized PyTorch Transformer models, which can be converted to LiteRT-LM models,
allowing users to easily deploy Large Language Models (LLMs) on edge
devices. Users can run the converted models via [LiteRT-LM](https://github.com/google-ai-edge/LiteRT-LM).

*Tip: When working with the Generative API, you can package your converted `.tflite` model alongside a tokenizer into a deployment-ready `.litertlm` container using the `litert-lm-builder` CLI tool (installed via the nightly package).*

More detailed documentation can be found [here](litert_torch/generative).

The Generative API currently supports CPU and GPU, with planned support for NPU.
A further future direction is to collaborate with the PyTorch community to
ensure that frequently used transformer abstractions can be directly supported
without reauthoring.

## Build Status

Build Type         |    Status     |
-----------        | --------------|
Generative API (Linux) | [![](https://github.com/google-ai-edge/litert-torch/actions/workflows/nightly_generative_api.yml/badge.svg?branch=main)](https://github.com/google-ai-edge/litert-torch/actions/workflows/nightly_generative_api.yml) |
Model Coverage (Linux) | [![](https://github.com/google-ai-edge/litert-torch/actions/workflows/nightly_model_coverage.yml/badge.svg?branch=main)](https://github.com/google-ai-edge/litert-torch/actions/workflows/nightly_model_coverage.yml) |
Unit tests (Linux)     | [![](https://github.com/google-ai-edge/litert-torch/actions/workflows/nightly_unittests.yml/badge.svg?branch=main)](https://github.com/google-ai-edge/litert-torch/actions/workflows/nightly_unittests.yml) |
Nightly Release    | [![](https://github.com/google-ai-edge/litert-torch/actions/workflows/nightly_release.yml/badge.svg?branch=main)](https://github.com/google-ai-edge/litert-torch/actions/workflows/nightly_release.yml) |

# Contributing

See our [contribution documentation](CONTRIBUTING.md).

# Getting Help

Please [create a GitHub issue](https://github.com/google-ai-edge/litert-torch/issues/new/choose) with any questions.