# Copyright (c) Chris Choy (chrischoy@ai.stanford.edu).
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
# of the Software, and to permit persons to whom the Software is furnished to do
# so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# Please cite "4D Spatio-Temporal ConvNets: Minkowski Convolutional Neural
# Networks", CVPR'19 (https://arxiv.org/abs/1904.08755) if you use any part
# of the code.
import torch
import torch.nn as nn
from torch.optim import SGD
from torchvision import transforms

import MinkowskiEngine as ME
# from modules.resnet_block import BasicBlock, Bottleneck
from model.modules.resnet_block import BasicBlock, Bottleneck, BasicBlockIN, BottleneckIN, BasicBlockLN

from collections import namedtuple

import numpy as np
from IPython import embed

class ResNetBaseEncoder(nn.Module):
    """ ResNetBase Encoder. """
    BLOCK = None
    LAYERS = ()
    INIT_DIM = 64
    PLANES = (64, 128, 256, 512)
    # OUT_TENSOR_STRIDE = 32

    def __init__(self, in_channels, D=3, width=64):
        nn.Module.__init__(self)
        self.D = D
        assert self.BLOCK is not None
        self.INIT_DIM = width
        self.PLANES = (width, width*2, width*4, width*8)

        self.network_initialization(in_channels, D)
        self.weight_initialization()


    def network_initialization(self, in_channels, D):

        self.inplanes = self.INIT_DIM
        self.conv1 = ME.MinkowskiConvolution(
            in_channels, self.inplanes, kernel_size=5, stride=1, dimension=D)

        self.bn1 = ME.MinkowskiBatchNorm(self.inplanes)
        self.relu = ME.MinkowskiReLU(inplace=True)
        self.pool = ME.MinkowskiSumPooling(kernel_size=2, stride=2, dimension=D)
        self.layer1 = self._make_layer(
            self.BLOCK, self.PLANES[0], self.LAYERS[0], stride=2)
        self.layer2 = self._make_layer(
            self.BLOCK, self.PLANES[1], self.LAYERS[1], stride=2)
        self.layer3 = self._make_layer(
            self.BLOCK, self.PLANES[2], self.LAYERS[2], stride=2)
        self.layer4 = self._make_layer(
            self.BLOCK, self.PLANES[3], self.LAYERS[3], stride=2)

        self.global_avg = ME.MinkowskiGlobalPooling(dimension=D)

        self.out_channels = self.inplanes

    def weight_initialization(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self,
                    block,
                    planes,
                    blocks,
                    stride=1,
                    dilation=1,
                    bn_momentum=0.1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                ME.MinkowskiConvolution(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    dimension=self.D),
                ME.MinkowskiBatchNorm(planes * block.expansion))
        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                stride=stride,
                dilation=dilation,
                downsample=downsample,
                D=self.D))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    stride=1,
                    dilation=dilation,
                    D=self.D))

        return nn.Sequential(*layers)
    

    def forward(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool(x)

        output1 = self.layer1(x)
        output2 = self.layer2(output1)
        output3 = self.layer3(output2)
        output4 = self.layer4(output3)

        output = self.global_avg(output4)

        return {
                "layer0": x,
                "layer1": output1,
                "layer2": output2,
                "layer3": output3,
                "layer4": output4,
                "output": output,
        }


class ResNet14Encoder(ResNetBaseEncoder):
    BLOCK = BasicBlock
    LAYERS = (1, 1, 1, 1)


class ResNet18Encoder(ResNetBaseEncoder):
    BLOCK = BasicBlock
    LAYERS = (2, 2, 2, 2)


class ResNet34Encoder(ResNetBaseEncoder):
    BLOCK = BasicBlock
    LAYERS = (3, 4, 6, 3)


class ResNet50Encoder(ResNetBaseEncoder):
    BLOCK = Bottleneck
    LAYERS = (3, 4, 6, 3)


class ResNet101Encoder(ResNetBaseEncoder):
    BLOCK = Bottleneck
    LAYERS = (3, 4, 23, 3)


def build_resnet_encoder(model_type):
    if model_type == "resnet14":
        return ResNet14Encoder
    elif model_type == "resnet18":
        return ResNet18Encoder
    elif model_type == "resnet34":
        return ResNet34Encoder
    elif model_type == "resnet50":
        return ResNet50Encoder
    elif model_type == "resnet101":
        return ResNet101Encoder
    else:
        raise NotImplementedError

