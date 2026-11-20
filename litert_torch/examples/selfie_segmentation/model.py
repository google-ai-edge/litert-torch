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
"""MediaPipe Selfie segmentation model ported to PyTorch.

# pylint: disable=line-too-long
First published in:
-
https://ai.google.dev/edge/mediapipe/solutions/vision/image_segmenter#selfie-model

Model Card
-
https://storage.googleapis.com/mediapipe-assets/Model%20Card%20MediaPipe%20Selfie%20Segmentation.pdf
# pylint: enable=line-too-long
"""

import torch
from torch import nn


def _DepthwiseConv2D(**kwargs):
  """Short-cut for creating a depthwise convolution."""
  return nn.Conv2d(groups=kwargs["in_channels"], **kwargs)


class SelfieSegmentation(nn.Module):
  """Selfie segmentation model."""

  def __init__(self):
    super(SelfieSegmentation, self).__init__()
    self.conv2d = nn.Conv2d(
        in_channels=3,
        out_channels=16,
        kernel_size=(3, 3),
        stride=(2, 2),
        padding=1,
    )
    self.hardswish = nn.Hardswish()
    self.conv2d_1 = nn.Conv2d(
        in_channels=16,
        out_channels=16,
        kernel_size=(1, 1),
    )
    self.relu = nn.ReLU()
    self.depthwise_conv2d = _DepthwiseConv2D(
        in_channels=16,
        out_channels=16,
        kernel_size=(3, 3),
        stride=(2, 2),
        padding=1,
    )
    self.average_pooling2d = nn.AvgPool2d(
        kernel_size=(64, 64),
        stride=(64, 64),
        padding=0,
    )
    self.conv2d_2 = nn.Conv2d(
        in_channels=16,
        out_channels=8,
        kernel_size=(1, 1),
        stride=(1, 1),
        padding=0,
    )
    self.conv2d_3 = nn.Conv2d(
        in_channels=8,
        out_channels=16,
        kernel_size=(1, 1),
        stride=(1, 1),
        padding=0,
    )
    self.activation = nn.Sigmoid()
    self.conv2d_4 = nn.Conv2d(
        in_channels=16,
        out_channels=16,
        kernel_size=(1, 1),
    )
    self.conv2d_5 = nn.Conv2d(
        in_channels=16,
        out_channels=72,
        kernel_size=(1, 1),
    )
    self.depthwise_conv2d_1 = _DepthwiseConv2D(
        in_channels=72,
        out_channels=72,
        kernel_size=(3, 3),
        stride=(2, 2),
        padding=1,
    )
    self.conv2d_6 = nn.Conv2d(
        in_channels=72,
        out_channels=24,
        kernel_size=(1, 1),
    )
    self.conv2d_7 = nn.Conv2d(
        in_channels=24,
        out_channels=88,
        kernel_size=(1, 1),
    )
    self.depthwise_conv2d_2 = _DepthwiseConv2D(
        in_channels=88,
        out_channels=88,
        kernel_size=(3, 3),
        stride=(1, 1),
        padding=1,
    )
    self.conv2d_8 = nn.Conv2d(
        in_channels=88,
        out_channels=24,
        kernel_size=(1, 1),
    )
    self.conv2d_9 = nn.Conv2d(
        in_channels=24,
        out_channels=96,
        kernel_size=(1, 1),
        stride=(1, 1),
        padding=0,
    )
    self.depthwise_conv2d_3 = _DepthwiseConv2D(
        in_channels=96,
        out_channels=96,
        kernel_size=(5, 5),
        stride=(2, 2),
        padding=2,
    )
    self.average_pooling2d_1 = nn.AvgPool2d(
        kernel_size=(16, 16), stride=(16, 16), padding=0
    )
    self.conv2d_10 = nn.Conv2d(
        in_channels=96,
        out_channels=24,
        kernel_size=(1, 1),
        stride=(1, 1),
        padding=0,
    )
    self.conv2d_11 = nn.Conv2d(
        in_channels=24,
        out_channels=96,
        kernel_size=(1, 1),
        stride=(1, 1),
        padding=0,
    )
    self.conv2d_12 = nn.Conv2d(
        in_channels=96,
        out_channels=32,
        kernel_size=(1, 1),
    )
    self.conv2d_13 = nn.Conv2d(
        in_channels=32,
        out_channels=128,
        kernel_size=(1, 1),
        stride=(1, 1),
        padding=0,
    )
    self.depthwise_conv2d_4 = _DepthwiseConv2D(
        in_channels=128,
        out_channels=128,
        kernel_size=(5, 5),
        stride=(1, 1),
        padding=2,
    )
    self.average_pooling2d_2 = nn.AvgPool2d(
        kernel_size=(16, 16), stride=(16, 16), padding=0
    )
    self.conv2d_14 = nn.Conv2d(
        in_channels=128,
        out_channels=32,
        kernel_size=(1, 1),
        stride=(1, 1),
        padding=0,
    )
    self.conv2d_15 = nn.Conv2d(
        in_channels=32,
        out_channels=128,
        kernel_size=(1, 1),
        stride=(1, 1),
        padding=0,
    )
    self.conv2d_16 = nn.Conv2d(
        in_channels=128,
        out_channels=32,
        kernel_size=(1, 1),
    )
    self.conv2d_17 = nn.Conv2d(
        in_channels=32,
        out_channels=128,
        kernel_size=(1, 1),
        stride=(1, 1),
        padding=0,
    )
    self.depthwise_conv2d_5 = _DepthwiseConv2D(
        in_channels=128,
        out_channels=128,
        kernel_size=(5, 5),
        stride=(1, 1),
        padding=2,
    )
    self.average_pooling2d_3 = nn.AvgPool2d(
        kernel_size=(16, 16), stride=(16, 16), padding=0
    )
    self.conv2d_18 = nn.Conv2d(
        in_channels=128,
        out_channels=32,
        kernel_size=(1, 1),
        stride=(1, 1),
        padding=0,
    )
    self.conv2d_19 = nn.Conv2d(
        in_channels=32,
        out_channels=128,
        kernel_size=(1, 1),
        stride=(1, 1),
        padding=0,
    )
    self.conv2d_20 = nn.Conv2d(
        in_channels=128,
        out_channels=32,
        kernel_size=(1, 1),
    )
    self.conv2d_21 = nn.Conv2d(
        in_channels=32,
        out_channels=96,
        kernel_size=(1, 1),
        stride=(1, 1),
        padding=0,
    )
    self.depthwise_conv2d_6 = _DepthwiseConv2D(
        in_channels=96,
        out_channels=96,
        kernel_size=(5, 5),
        stride=(1, 1),
        padding=2,
    )
    self.average_pooling2d_4 = nn.AvgPool2d(
        kernel_size=(16, 16), stride=(16, 16), padding=0
    )
    self.conv2d_22 = nn.Conv2d(
        in_channels=96,
        out_channels=24,
        kernel_size=(1, 1),
        stride=(1, 1),
        padding=0,
    )
    self.conv2d_23 = nn.Conv2d(
        in_channels=24,
        out_channels=96,
        kernel_size=(1, 1),
        stride=(1, 1),
        padding=0,
    )
    self.conv2d_24 = nn.Conv2d(
        in_channels=96,
        out_channels=32,
        kernel_size=(1, 1),
        stride=(1, 1),
        padding=0,
    )
    self.conv2d_25 = nn.Conv2d(
        in_channels=32,
        out_channels=96,
        kernel_size=(1, 1),
        stride=(1, 1),
        padding=0,
    )
    self.depthwise_conv2d_7 = _DepthwiseConv2D(
        in_channels=96,
        out_channels=96,
        kernel_size=(5, 5),
        stride=(1, 1),
        padding=2,
    )
    self.average_pooling2d_5 = nn.AvgPool2d(
        kernel_size=(16, 16), stride=(16, 16), padding=0
    )
    self.conv2d_26 = nn.Conv2d(
        in_channels=96,
        out_channels=24,
        kernel_size=(1, 1),
        stride=(1, 1),
        padding=0,
    )
    self.conv2d_27 = nn.Conv2d(
        in_channels=24,
        out_channels=96,
        kernel_size=(1, 1),
        stride=(1, 1),
        padding=0,
    )
    self.conv2d_28 = nn.Conv2d(
        in_channels=96,
        out_channels=32,
        kernel_size=(1, 1),
        stride=(1, 1),
        padding=0,
    )
    self.average_pooling2d_6 = nn.AvgPool2d(
        kernel_size=(16, 16), stride=(16, 16), padding=0
    )
    self.conv2d_29 = nn.Conv2d(
        in_channels=32,
        out_channels=128,
        kernel_size=(1, 1),
        stride=(1, 1),
        padding=0,
    )
    self.conv2d_30 = nn.Conv2d(
        in_channels=32,
        out_channels=128,
        kernel_size=(1, 1),
        stride=(1, 1),
        padding=0,
    )
    self.conv2d_31 = nn.Conv2d(
        in_channels=128,
        out_channels=24,
        kernel_size=(1, 1),
        stride=(1, 1),
        padding=0,
    )
    self.average_pooling2d_7 = nn.AvgPool2d(
        kernel_size=(32, 32), stride=(32, 32), padding=0
    )
    self.conv2d_32 = nn.Conv2d(
        in_channels=24,
        out_channels=24,
        kernel_size=(1, 1),
        stride=(1, 1),
        padding=0,
    )
    self.conv2d_33 = nn.Conv2d(
        in_channels=24,
        out_channels=24,
        kernel_size=(1, 1),
        stride=(1, 1),
        padding=0,
    )
    self.conv2d_34 = nn.Conv2d(
        in_channels=24,
        out_channels=24,
        kernel_size=(1, 1),
        stride=(1, 1),
        padding=0,
    )
    self.depthwise_conv2d_8 = _DepthwiseConv2D(
        in_channels=24,
        out_channels=24,
        kernel_size=(3, 3),
        stride=(1, 1),
        padding=1,
    )
    self.conv2d_35 = nn.Conv2d(
        in_channels=24,
        out_channels=16,
        kernel_size=(1, 1),
        stride=(1, 1),
        padding=0,
    )
    self.average_pooling2d_8 = nn.AvgPool2d(
        kernel_size=(64, 64), stride=(64, 64), padding=0
    )
    self.conv2d_36 = nn.Conv2d(
        in_channels=16,
        out_channels=16,
        kernel_size=(1, 1),
        stride=(1, 1),
        padding=0,
    )
    self.conv2d_37 = nn.Conv2d(
        in_channels=16,
        out_channels=16,
        kernel_size=(1, 1),
        stride=(1, 1),
        padding=0,
    )
    self.conv2d_38 = nn.Conv2d(
        in_channels=16,
        out_channels=16,
        kernel_size=(1, 1),
        stride=(1, 1),
        padding=0,
    )
    self.depthwise_conv2d_9 = _DepthwiseConv2D(
        in_channels=16,
        out_channels=16,
        kernel_size=(3, 3),
        stride=(1, 1),
        padding=1,
    )
    self.conv2d_39 = nn.Conv2d(
        in_channels=16,
        out_channels=16,
        kernel_size=(1, 1),
        stride=(1, 1),
        padding=0,
    )
    self.average_pooling2d_9 = nn.AvgPool2d(
        kernel_size=(128, 128), stride=(128, 128), padding=0
    )
    self.conv2d_40 = nn.Conv2d(
        in_channels=16,
        out_channels=16,
        kernel_size=(1, 1),
        stride=(1, 1),
        padding=0,
    )
    self.conv2d_41 = nn.Conv2d(
        in_channels=16,
        out_channels=16,
        kernel_size=(1, 1),
        stride=(1, 1),
        padding=0,
    )
    self.conv2d_42 = nn.Conv2d(
        in_channels=16,
        out_channels=16,
        kernel_size=(1, 1),
        stride=(1, 1),
        padding=0,
    )
    self.depthwise_conv2d_10 = _DepthwiseConv2D(
        in_channels=16,
        out_channels=16,
        kernel_size=(3, 3),
        stride=(1, 1),
        padding=1,
    )
    self.segment = nn.ConvTranspose2d(16, 1, 2, 2, 0)
    self.up_sampling2d = nn.Upsample(scale_factor=2, mode="bilinear")

  def forward(self, image):
    conv2d = self.conv2d(image)
    h_swish = self.hardswish(conv2d)
    conv2d_1 = self.conv2d_1(h_swish)
    re_lu = self.relu(conv2d_1)
    depthwise_conv2d = self.depthwise_conv2d(re_lu)
    re_lu_1 = self.relu(depthwise_conv2d)
    average_pooling2d = self.average_pooling2d(re_lu_1)
    conv2d_2 = self.conv2d_2(average_pooling2d)
    re_lu_2 = self.relu(conv2d_2)
    conv2d_3 = self.conv2d_3(re_lu_2)
    activation = self.activation(conv2d_3)
    multiply = re_lu_1 * activation
    conv2d_4 = self.conv2d_4(multiply)
    conv2d_5 = self.conv2d_5(conv2d_4)
    re_lu_3 = self.relu(conv2d_5)
    depthwise_conv2d_1 = self.depthwise_conv2d_1(re_lu_3)
    re_lu_4 = self.relu(depthwise_conv2d_1)
    conv2d_6 = self.conv2d_6(re_lu_4)
    conv2d_7 = self.conv2d_7(conv2d_6)
    re_lu_5 = self.relu(conv2d_7)
    depthwise_conv2d_2 = self.depthwise_conv2d_2(re_lu_5)
    re_lu_6 = self.relu(depthwise_conv2d_2)
    conv2d_8 = self.conv2d_8(re_lu_6)
    add = conv2d_8 + conv2d_6
    conv2d_9 = self.conv2d_9(add)
    h_swish_1 = self.hardswish(conv2d_9)
    depthwise_conv2d_3 = self.depthwise_conv2d_3(h_swish_1)
    h_swish_2 = self.hardswish(depthwise_conv2d_3)
    average_pooling2d_1 = self.average_pooling2d_1(h_swish_2)
    conv2d_10 = self.conv2d_10(average_pooling2d_1)
    re_lu_7 = self.relu(conv2d_10)
    conv2d_11 = self.conv2d_11(re_lu_7)
    activation_1 = self.activation(conv2d_11)
    multiply_1 = h_swish_2 * activation_1
    conv2d_12 = self.conv2d_12(multiply_1)
    conv2d_13 = self.conv2d_13(conv2d_12)
    h_swish_3 = self.hardswish(conv2d_13)
    depthwise_conv2d_4 = self.depthwise_conv2d_4(h_swish_3)
    h_swish_4 = self.hardswish(depthwise_conv2d_4)
    average_pooling2d_2 = self.average_pooling2d_2(h_swish_4)
    conv2d_14 = self.conv2d_14(average_pooling2d_2)
    re_lu_8 = self.relu(conv2d_14)
    conv2d_15 = self.conv2d_15(re_lu_8)
    activation_2 = self.activation(conv2d_15)
    multiply_2 = h_swish_4 * activation_2
    conv2d_16 = self.conv2d_16(multiply_2)
    add_1 = conv2d_16 + conv2d_12
    conv2d_17 = self.conv2d_17(add_1)
    h_swish_5 = self.hardswish(conv2d_17)
    depthwise_conv2d_5 = self.depthwise_conv2d_5(h_swish_5)
    h_swish_6 = self.hardswish(depthwise_conv2d_5)
    average_pooling2d_3 = self.average_pooling2d_3(h_swish_6)
    conv2d_18 = self.conv2d_18(average_pooling2d_3)
    re_lu_9 = self.relu(conv2d_18)
    conv2d_19 = self.conv2d_19(re_lu_9)
    activation_3 = self.activation(conv2d_19)
    multiply_3 = h_swish_6 * activation_3
    conv2d_20 = self.conv2d_20(multiply_3)
    add_2 = conv2d_20 + add_1
    conv2d_21 = self.conv2d_21(add_2)
    h_swish_7 = self.hardswish(conv2d_21)
    depthwise_conv2d_6 = self.depthwise_conv2d_6(h_swish_7)
    h_swish_8 = self.hardswish(depthwise_conv2d_6)
    average_pooling2d_4 = self.average_pooling2d_4(h_swish_8)
    conv2d_22 = self.conv2d_22(average_pooling2d_4)
    re_lu_10 = self.relu(conv2d_22)
    conv2d_23 = self.conv2d_23(re_lu_10)
    activation_4 = self.activation(conv2d_23)
    multiply_4 = h_swish_8 * activation_4
    conv2d_24 = self.conv2d_24(multiply_4)
    add_3 = conv2d_24 + add_2
    conv2d_25 = self.conv2d_25(add_3)
    h_swish_9 = self.hardswish(conv2d_25)
    depthwise_conv2d_7 = self.depthwise_conv2d_7(h_swish_9)
    h_swish_10 = self.hardswish(depthwise_conv2d_7)
    average_pooling2d_5 = self.average_pooling2d_5(h_swish_10)
    conv2d_26 = self.conv2d_26(average_pooling2d_5)
    re_lu_11 = self.relu(conv2d_26)
    conv2d_27 = self.conv2d_27(re_lu_11)
    activation_5 = self.activation(conv2d_27)
    multiply_5 = h_swish_10 * activation_5
    conv2d_28 = self.conv2d_28(multiply_5)
    add_4 = conv2d_28 + add_3
    average_pooling2d_6 = self.average_pooling2d_6(add_4)
    conv2d_29 = self.conv2d_29(add_4)
    conv2d_30 = self.conv2d_30(average_pooling2d_6)
    re_lu_12 = self.relu(conv2d_29)
    activation_6 = self.activation(conv2d_30)
    multiply_6 = re_lu_12 * activation_6
    up_sampling2d = self.up_sampling2d(multiply_6)
    conv2d_31 = self.conv2d_31(up_sampling2d)
    add_5 = add + conv2d_31
    average_pooling2d_7 = self.average_pooling2d_7(add_5)
    conv2d_32 = self.conv2d_32(average_pooling2d_7)
    re_lu_13 = self.relu(conv2d_32)
    conv2d_33 = self.conv2d_33(re_lu_13)
    activation_7 = self.activation(conv2d_33)
    multiply_7 = add * activation_7
    add_6 = multiply_7 + conv2d_31
    conv2d_34 = self.conv2d_34(add_6)
    re_lu_14 = self.relu(conv2d_34)
    depthwise_conv2d_8 = self.depthwise_conv2d_8(re_lu_14)
    re_lu_15 = self.relu(depthwise_conv2d_8)
    add_7 = re_lu_14 + re_lu_15
    up_sampling2d_1 = self.up_sampling2d(add_7)
    conv2d_35 = self.conv2d_35(up_sampling2d_1)
    add_8 = conv2d_4 + conv2d_35
    average_pooling2d_8 = self.average_pooling2d_8(add_8)
    conv2d_36 = self.conv2d_36(average_pooling2d_8)
    re_lu_16 = self.relu(conv2d_36)
    conv2d_37 = self.conv2d_37(re_lu_16)
    activation_8 = self.activation(conv2d_37)
    multiply_8 = conv2d_4 + activation_8
    add_9 = multiply_8 + conv2d_35
    conv2d_38 = self.conv2d_38(add_9)
    re_lu_17 = self.relu(conv2d_38)
    depthwise_conv2d_9 = self.depthwise_conv2d_9(re_lu_17)
    re_lu_18 = self.relu(depthwise_conv2d_9)
    add_10 = re_lu_17 + re_lu_18
    up_sampling2d_2 = self.up_sampling2d(add_10)
    conv2d_39 = self.conv2d_39(up_sampling2d_2)
    add_11 = h_swish + conv2d_39
    average_pooling2d_9 = self.average_pooling2d_9(add_11)
    conv2d_40 = self.conv2d_40(average_pooling2d_9)
    re_lu_19 = self.relu(conv2d_40)
    conv2d_41 = self.conv2d_41(re_lu_19)
    activation_9 = self.activation(conv2d_41)
    multiply_9 = h_swish * activation_9
    add_12 = multiply_9 + conv2d_39
    conv2d_42 = self.conv2d_42(add_12)
    re_lu_20 = self.relu(conv2d_42)
    depthwise_conv2d_10 = self.depthwise_conv2d_10(re_lu_20)
    re_lu_21 = self.relu(depthwise_conv2d_10)
    add_13 = re_lu_20 + re_lu_21
    segment = self.segment(add_13)
    return self.activation(segment)

  def load_from_pth(self, pth_path: str):
    """Loads the model from a pth file.

    Args:
      pth_path: The path to the pth file to load the model from.
    """
    self.load_state_dict(torch.load(pth_path, weights_only=True))
