# Copyright 2024 The LiteRT Torch Authors.
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

import os
import pathlib
import re

import setuptools

setup = setuptools.setup
find_packages = setuptools.find_packages
here = pathlib.Path(__file__).parent.resolve()

# Get the long description from the README file
long_description = """
Library that supports converting PyTorch models into a .tflite format, which can
then be run with LiteRT.  This enables applications for
Android, iOS and IOT that can run models completely on-device.

[Install steps](https://github.com/google-ai-edge/litert-torch#installation)
and additional details are in the LiteRT Torch
[GitHub repository](https://github.com/google-ai-edge/litert-torch).
""".lstrip()

name = "litert-torch"
# TODO(b/357076369): move version updating logics to version.py
version_py = here / "litert_torch" / "version.py"
version_regex = "__version__\s*=\s*(\"|')(?P<version>[^\"']+)(\"|')"
version = re.search(version_regex, version_py.read_text()).group("version")

if nightly_release_date := os.environ.get("NIGHTLY_RELEASE_DATE"):
  name += "-nightly"
  version += ".dev" + nightly_release_date
  version_py.write_text(
      re.sub(
          version_regex, f'__version__ = "{version}"', version_py.read_text()
      )
  )

setup(
    name=name,
    version=version,
    description="Support PyTorch model conversion with LiteRT.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/google-ai-edge/litert-torch",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development",
        "Topic :: Software Development :: Libraries",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    keywords="On-Device ML, AI, Google, TFLite, LiteRT, PyTorch, LLMs, GenAI",
    packages=find_packages(
        include=["litert_torch*"],
    ),
    package_data={
        "litert_torch.generative.export_hf.experimental.npu_export": [
            "*.json",
            "configs/*.json",
        ],
    },
    python_requires=">=3.10",
    install_requires=[
        "absl-py",
        "numpy",
        "scipy",
        "safetensors",
        "multipledispatch",
        "transformers",
        "kagglehub",
        "tabulate",
        "torch>=2.4.0,<2.14.0",
        "ai-edge-litert[model-utils]>=2.2.0,<2.3.0",
        "ai-edge-quantizer==0.9.*",
        "litert-converter==0.4.*",
        "torchao>=0.17.0",
        "jax",
        "jaxtyping",
        "fire",
        "sentencepiece",
        "rich",
        "litert-lm-builder>=0.0.0.dev0",
        "pillow",
    ],
    entry_points={
        "console_scripts": [
            "litert-torch = litert_torch.cli:main",
        ],
    },
)
