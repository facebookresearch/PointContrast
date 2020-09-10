from model.resnet import ResNetBase, get_norm
from model.modules.common import ConvType, NormType, conv, conv_tr
from model.modules.resnet_block import BasicBlock, Bottleneck, BasicBlockIN, BottleneckIN, BasicBlockLN

from MinkowskiEngine import MinkowskiReLU, MinkowskiGlobalPooling
from MinkowskiEngine import SparseTensor
import MinkowskiEngine.MinkowskiOps as me

import torch
import torch.nn as nn

class Res16UNetBase(ResNetBase):
  BLOCK = None
  PLANES = (32, 64, 128, 256, 256, 256, 256, 256)
  DILATIONS = (1, 1, 1, 1, 1, 1, 1, 1)
  LAYERS = (2, 2, 2, 2, 2, 2, 2, 2)
  INIT_DIM = 32
  OUT_PIXEL_DIST = 1
  NORM_TYPE = NormType.BATCH_NORM
  NON_BLOCK_CONV_TYPE = ConvType.SPATIAL_HYPERCUBE
  CONV_TYPE = ConvType.SPATIAL_HYPERCUBE_TEMPORAL_HYPERCROSS

  # To use the model, must call initialize_coords before forward pass.
  # Once data is processed, call clear to reset the model before calling initialize_coords
  def __init__(self,
               in_channels,
               out_channels,
               config,
               D=3):
    super(Res16UNetBase, self).__init__(in_channels, out_channels, config, D)
    self.normalize_feature = config.net.normalize_feature

  def network_initialization(self, in_channels, out_channels, config, D):
    # Setup net_metadata
    dilations = self.DILATIONS
    bn_momentum = config.opt.bn_momentum

    def space_n_time_m(n, m):
      return n if D == 3 else [n, n, n, m]

    if D == 4:
      self.OUT_PIXEL_DIST = space_n_time_m(self.OUT_PIXEL_DIST, 1)

    # Output of the first conv concated to conv6
    self.inplanes = self.INIT_DIM
    self.conv0p1s1 = conv(
        in_channels,
        self.inplanes,
        kernel_size=space_n_time_m(config.net.conv1_kernel_size, 1),
        stride=1,
        dilation=1,
        conv_type=self.NON_BLOCK_CONV_TYPE,
        D=D)

    self.bn0 = get_norm(self.NORM_TYPE, self.inplanes, D, bn_momentum=bn_momentum)

    self.conv1p1s2 = conv(
        self.inplanes,
        self.inplanes,
        kernel_size=space_n_time_m(2, 1),
        stride=space_n_time_m(2, 1),
        dilation=1,
        conv_type=self.NON_BLOCK_CONV_TYPE,
        D=D)
    self.bn1 = get_norm(self.NORM_TYPE, self.inplanes, D, bn_momentum=bn_momentum)
    self.block1 = self._make_layer(
        self.BLOCK,
        self.PLANES[0],
        self.LAYERS[0],
        dilation=dilations[0],
        norm_type=self.NORM_TYPE,
        bn_momentum=bn_momentum)

    self.conv2p2s2 = conv(
        self.inplanes,
        self.inplanes,
        kernel_size=space_n_time_m(2, 1),
        stride=space_n_time_m(2, 1),
        dilation=1,
        conv_type=self.NON_BLOCK_CONV_TYPE,
        D=D)
    self.bn2 = get_norm(self.NORM_TYPE, self.inplanes, D, bn_momentum=bn_momentum)
    self.block2 = self._make_layer(
        self.BLOCK,
        self.PLANES[1],
        self.LAYERS[1],
        dilation=dilations[1],
        norm_type=self.NORM_TYPE,
        bn_momentum=bn_momentum)

    self.conv3p4s2 = conv(
        self.inplanes,
        self.inplanes,
        kernel_size=space_n_time_m(2, 1),
        stride=space_n_time_m(2, 1),
        dilation=1,
        conv_type=self.NON_BLOCK_CONV_TYPE,
        D=D)
    self.bn3 = get_norm(self.NORM_TYPE, self.inplanes, D, bn_momentum=bn_momentum)
    self.block3 = self._make_layer(
        self.BLOCK,
        self.PLANES[2],
        self.LAYERS[2],
        dilation=dilations[2],
        norm_type=self.NORM_TYPE,
        bn_momentum=bn_momentum)

    self.conv4p8s2 = conv(
        self.inplanes,
        self.inplanes,
        kernel_size=space_n_time_m(2, 1),
        stride=space_n_time_m(2, 1),
        dilation=1,
        conv_type=self.NON_BLOCK_CONV_TYPE,
        D=D)
    self.bn4 = get_norm(self.NORM_TYPE, self.inplanes, D, bn_momentum=bn_momentum)
    self.block4 = self._make_layer(
        self.BLOCK,
        self.PLANES[3],
        self.LAYERS[3],
        dilation=dilations[3],
        norm_type=self.NORM_TYPE,
        bn_momentum=bn_momentum)
    self.convtr4p16s2 = conv_tr(
        self.inplanes,
        self.PLANES[4],
        kernel_size=space_n_time_m(2, 1),
        upsample_stride=space_n_time_m(2, 1),
        dilation=1,
        bias=False,
        conv_type=self.NON_BLOCK_CONV_TYPE,
        D=D)
    self.bntr4 = get_norm(self.NORM_TYPE, self.PLANES[4], D, bn_momentum=bn_momentum)

    self.inplanes = self.PLANES[4] + self.PLANES[2] * self.BLOCK.expansion
    self.block5 = self._make_layer(
        self.BLOCK,
        self.PLANES[4],
        self.LAYERS[4],
        dilation=dilations[4],
        norm_type=self.NORM_TYPE,
        bn_momentum=bn_momentum)
    self.convtr5p8s2 = conv_tr(
        self.inplanes,
        self.PLANES[5],
        kernel_size=space_n_time_m(2, 1),
        upsample_stride=space_n_time_m(2, 1),
        dilation=1,
        bias=False,
        conv_type=self.NON_BLOCK_CONV_TYPE,
        D=D)
    self.bntr5 = get_norm(self.NORM_TYPE, self.PLANES[5], D, bn_momentum=bn_momentum)

    self.inplanes = self.PLANES[5] + self.PLANES[1] * self.BLOCK.expansion
    self.block6 = self._make_layer(
        self.BLOCK,
        self.PLANES[5],
        self.LAYERS[5],
        dilation=dilations[5],
        norm_type=self.NORM_TYPE,
        bn_momentum=bn_momentum)
    self.convtr6p4s2 = conv_tr(
        self.inplanes,
        self.PLANES[6],
        kernel_size=space_n_time_m(2, 1),
        upsample_stride=space_n_time_m(2, 1),
        dilation=1,
        bias=False,
        conv_type=self.NON_BLOCK_CONV_TYPE,
        D=D)
    self.bntr6 = get_norm(self.NORM_TYPE, self.PLANES[6], D, bn_momentum=bn_momentum)

    self.inplanes = self.PLANES[6] + self.PLANES[0] * self.BLOCK.expansion
    self.block7 = self._make_layer(
        self.BLOCK,
        self.PLANES[6],
        self.LAYERS[6],
        dilation=dilations[6],
        norm_type=self.NORM_TYPE,
        bn_momentum=bn_momentum)
    self.convtr7p2s2 = conv_tr(
        self.inplanes,
        self.PLANES[7],
        kernel_size=space_n_time_m(2, 1),
        upsample_stride=space_n_time_m(2, 1),
        dilation=1,
        bias=False,
        conv_type=self.NON_BLOCK_CONV_TYPE,
        D=D)
    self.bntr7 = get_norm(self.NORM_TYPE, self.PLANES[7], D, bn_momentum=bn_momentum)

    self.inplanes = self.PLANES[7] + self.INIT_DIM
    self.block8 = self._make_layer(
        self.BLOCK,
        self.PLANES[7],
        self.LAYERS[7],
        dilation=dilations[7],
        norm_type=self.NORM_TYPE,
        bn_momentum=bn_momentum)

    self.final = conv(self.PLANES[7], out_channels, kernel_size=1, stride=1, bias=True, D=D)
    self.relu = MinkowskiReLU(inplace=True)

  def forward(self, x):
    out = self.conv0p1s1(x)
    out = self.bn0(out)
    out_p1 = self.relu(out)

    out = self.conv1p1s2(out_p1)
    out = self.bn1(out)
    out = self.relu(out)
    out_b1p2 = self.block1(out)

    out = self.conv2p2s2(out_b1p2)
    out = self.bn2(out)
    out = self.relu(out)
    out_b2p4 = self.block2(out)

    out = self.conv3p4s2(out_b2p4)
    out = self.bn3(out)
    out = self.relu(out)
    out_b3p8 = self.block3(out)

    # pixel_dist=16
    out = self.conv4p8s2(out_b3p8)
    out = self.bn4(out)
    out = self.relu(out)
    encoder_out = self.block4(out)
     
    # pixel_dist=8
    out = self.convtr4p16s2(encoder_out)
    out = self.bntr4(out)
    out = self.relu(out)

    out = me.cat(out, out_b3p8)
    out = self.block5(out)

    # pixel_dist=4
    out = self.convtr5p8s2(out)
    out = self.bntr5(out)
    out = self.relu(out)

    out = me.cat(out, out_b2p4)
    out = self.block6(out)

    # pixel_dist=2
    out = self.convtr6p4s2(out)
    out = self.bntr6(out)
    out = self.relu(out)

    out = me.cat(out, out_b1p2)
    out = self.block7(out)

    # pixel_dist=1
    out = self.convtr7p2s2(out)
    out = self.bntr7(out)
    out = self.relu(out)

    out = me.cat(out, out_p1)
    out = self.block8(out)

    out =  self.final(out)
        
    if self.normalize_feature:
      return SparseTensor(
          out.F / torch.norm(out.F, p=2, dim=1, keepdim=True),
          coords_key=out.coords_key,
          coords_manager=out.coords_man)
    else:
      return out

class MonitorRes16UNetBase(ResNetBase):
  BLOCK = None
  PLANES = (32, 64, 128, 256, 256, 256, 256, 256)
  DILATIONS = (1, 1, 1, 1, 1, 1, 1, 1)
  LAYERS = (2, 2, 2, 2, 2, 2, 2, 2)
  INIT_DIM = 32
  OUT_PIXEL_DIST = 1
  NORM_TYPE = NormType.BATCH_NORM
  NON_BLOCK_CONV_TYPE = ConvType.SPATIAL_HYPERCUBE
  CONV_TYPE = ConvType.SPATIAL_HYPERCUBE_TEMPORAL_HYPERCROSS

  # To use the model, must call initialize_coords before forward pass.
  # Once data is processed, call clear to reset the model before calling initialize_coords
  def __init__(self,
               in_channels,
               out_channels,
               config,
               D=3):
    super(MonitorRes16UNetBase, self).__init__(in_channels, out_channels, config, D)
    self.normalize_feature = config.normalize_feature

  def network_initialization(self, in_channels, out_channels, config, D):
    # Setup net_metadata
    dilations = self.DILATIONS
    bn_momentum = config.opt.bn_momentum

    def space_n_time_m(n, m):
      return n if D == 3 else [n, n, n, m]

    if D == 4:
      self.OUT_PIXEL_DIST = space_n_time_m(self.OUT_PIXEL_DIST, 1)

    # Output of the first conv concated to conv6
    self.inplanes = self.INIT_DIM
    self.conv0p1s1 = conv(
        in_channels,
        self.inplanes,
        kernel_size=space_n_time_m(config.conv1_kernel_size, 1),
        stride=1,
        dilation=1,
        conv_type=self.NON_BLOCK_CONV_TYPE,
        D=D)

    self.bn0 = get_norm(self.NORM_TYPE, self.inplanes, D, bn_momentum=bn_momentum)

    self.conv1p1s2 = conv(
        self.inplanes,
        self.inplanes,
        kernel_size=space_n_time_m(2, 1),
        stride=space_n_time_m(2, 1),
        dilation=1,
        conv_type=self.NON_BLOCK_CONV_TYPE,
        D=D)
    self.bn1 = get_norm(self.NORM_TYPE, self.inplanes, D, bn_momentum=bn_momentum)
    self.block1 = self._make_layer(
        self.BLOCK,
        self.PLANES[0],
        self.LAYERS[0],
        dilation=dilations[0],
        norm_type=self.NORM_TYPE,
        bn_momentum=bn_momentum)

    self.conv2p2s2 = conv(
        self.inplanes,
        self.inplanes,
        kernel_size=space_n_time_m(2, 1),
        stride=space_n_time_m(2, 1),
        dilation=1,
        conv_type=self.NON_BLOCK_CONV_TYPE,
        D=D)
    self.bn2 = get_norm(self.NORM_TYPE, self.inplanes, D, bn_momentum=bn_momentum)
    self.block2 = self._make_layer(
        self.BLOCK,
        self.PLANES[1],
        self.LAYERS[1],
        dilation=dilations[1],
        norm_type=self.NORM_TYPE,
        bn_momentum=bn_momentum)

    self.conv3p4s2 = conv(
        self.inplanes,
        self.inplanes,
        kernel_size=space_n_time_m(2, 1),
        stride=space_n_time_m(2, 1),
        dilation=1,
        conv_type=self.NON_BLOCK_CONV_TYPE,
        D=D)
    self.bn3 = get_norm(self.NORM_TYPE, self.inplanes, D, bn_momentum=bn_momentum)
    self.block3 = self._make_layer(
        self.BLOCK,
        self.PLANES[2],
        self.LAYERS[2],
        dilation=dilations[2],
        norm_type=self.NORM_TYPE,
        bn_momentum=bn_momentum)

    self.conv4p8s2 = conv(
        self.inplanes,
        self.inplanes,
        kernel_size=space_n_time_m(2, 1),
        stride=space_n_time_m(2, 1),
        dilation=1,
        conv_type=self.NON_BLOCK_CONV_TYPE,
        D=D)
    self.bn4 = get_norm(self.NORM_TYPE, self.inplanes, D, bn_momentum=bn_momentum)
    self.block4 = self._make_layer(
        self.BLOCK,
        self.PLANES[3],
        self.LAYERS[3],
        dilation=dilations[3],
        norm_type=self.NORM_TYPE,
        bn_momentum=bn_momentum)
    self.convtr4p16s2 = conv_tr(
        self.inplanes,
        self.PLANES[4],
        kernel_size=space_n_time_m(2, 1),
        upsample_stride=space_n_time_m(2, 1),
        dilation=1,
        bias=False,
        conv_type=self.NON_BLOCK_CONV_TYPE,
        D=D)
    self.bntr4 = get_norm(self.NORM_TYPE, self.PLANES[4], D, bn_momentum=bn_momentum)

    self.inplanes = self.PLANES[4] + self.PLANES[2] * self.BLOCK.expansion
    self.block5 = self._make_layer(
        self.BLOCK,
        self.PLANES[4],
        self.LAYERS[4],
        dilation=dilations[4],
        norm_type=self.NORM_TYPE,
        bn_momentum=bn_momentum)
    self.convtr5p8s2 = conv_tr(
        self.inplanes,
        self.PLANES[5],
        kernel_size=space_n_time_m(2, 1),
        upsample_stride=space_n_time_m(2, 1),
        dilation=1,
        bias=False,
        conv_type=self.NON_BLOCK_CONV_TYPE,
        D=D)
    self.bntr5 = get_norm(self.NORM_TYPE, self.PLANES[5], D, bn_momentum=bn_momentum)

    self.inplanes = self.PLANES[5] + self.PLANES[1] * self.BLOCK.expansion
    self.block6 = self._make_layer(
        self.BLOCK,
        self.PLANES[5],
        self.LAYERS[5],
        dilation=dilations[5],
        norm_type=self.NORM_TYPE,
        bn_momentum=bn_momentum)
    self.convtr6p4s2 = conv_tr(
        self.inplanes,
        self.PLANES[6],
        kernel_size=space_n_time_m(2, 1),
        upsample_stride=space_n_time_m(2, 1),
        dilation=1,
        bias=False,
        conv_type=self.NON_BLOCK_CONV_TYPE,
        D=D)
    self.bntr6 = get_norm(self.NORM_TYPE, self.PLANES[6], D, bn_momentum=bn_momentum)

    self.inplanes = self.PLANES[6] + self.PLANES[0] * self.BLOCK.expansion
    self.block7 = self._make_layer(
        self.BLOCK,
        self.PLANES[6],
        self.LAYERS[6],
        dilation=dilations[6],
        norm_type=self.NORM_TYPE,
        bn_momentum=bn_momentum)
    self.convtr7p2s2 = conv_tr(
        self.inplanes,
        self.PLANES[7],
        kernel_size=space_n_time_m(2, 1),
        upsample_stride=space_n_time_m(2, 1),
        dilation=1,
        bias=False,
        conv_type=self.NON_BLOCK_CONV_TYPE,
        D=D)
    self.bntr7 = get_norm(self.NORM_TYPE, self.PLANES[7], D, bn_momentum=bn_momentum)

    self.inplanes = self.PLANES[7] + self.INIT_DIM
    self.block8 = self._make_layer(
        self.BLOCK,
        self.PLANES[7],
        self.LAYERS[7],
        dilation=dilations[7],
        norm_type=self.NORM_TYPE,
        bn_momentum=bn_momentum)

    self.final = conv(self.PLANES[7], out_channels, kernel_size=1, stride=1, bias=True, D=D)
    self.relu = MinkowskiReLU(inplace=True)
    self.global_avg = MinkowskiGlobalPooling(dimension=D)


  def forward(self, x):
    out = self.conv0p1s1(x)
    out = self.bn0(out)
    out_p1 = self.relu(out)

    out = self.conv1p1s2(out_p1)
    out = self.bn1(out)
    out = self.relu(out)
    out_b1p2 = self.block1(out)

    out = self.conv2p2s2(out_b1p2)
    out = self.bn2(out)
    out = self.relu(out)
    out_b2p4 = self.block2(out)

    out = self.conv3p4s2(out_b2p4)
    out = self.bn3(out)
    out = self.relu(out)
    out_b3p8 = self.block3(out)

    # pixel_dist=16
    out = self.conv4p8s2(out_b3p8)
    out = self.bn4(out)
    out = self.relu(out)
    encoder_out = self.block4(out)
     
    # pixel_dist=8
    out = self.convtr4p16s2(encoder_out)
    out = self.bntr4(out)
    out = self.relu(out)

    out = me.cat(out, out_b3p8)
    out = self.block5(out)

    # pixel_dist=4
    out = self.convtr5p8s2(out)
    out = self.bntr5(out)
    out = self.relu(out)

    out = me.cat(out, out_b2p4)
    out = self.block6(out)

    # pixel_dist=2
    out = self.convtr6p4s2(out)
    out = self.bntr6(out)
    out = self.relu(out)

    out = me.cat(out, out_b1p2)
    out = self.block7(out)

    # pixel_dist=1
    out = self.convtr7p2s2(out)
    out = self.bntr7(out)
    out = self.relu(out)

    out = me.cat(out, out_p1)
    decoder_out = self.block8(out)

    out =  self.final(decoder_out)

    gap = self.global_avg(encoder_out)

    if self.normalize_feature:
      return SparseTensor(
          out.F / torch.norm(out.F, p=2, dim=1, keepdim=True),
          coords_key=out.coords_key,
          coords_manager=out.coords_man), head
    else:
      return out, head

class Res16UNet14(Res16UNetBase):
  BLOCK = BasicBlock
  LAYERS = (1, 1, 1, 1, 1, 1, 1, 1)


class Res16UNet18(Res16UNetBase):
  BLOCK = BasicBlock
  LAYERS = (2, 2, 2, 2, 2, 2, 2, 2)


class Res16UNet34(Res16UNetBase):
  BLOCK = BasicBlock
  LAYERS = (2, 3, 4, 6, 2, 2, 2, 2)

class MonitorRes16UNet34(MonitorRes16UNetBase):
  BLOCK = BasicBlock
  LAYERS = (2, 3, 4, 6, 2, 2, 2, 2)


class Res16UNet50(Res16UNetBase):
  BLOCK = Bottleneck
  LAYERS = (2, 3, 4, 6, 2, 2, 2, 2)


class Res16UNet101(Res16UNetBase):
  BLOCK = Bottleneck
  LAYERS = (2, 3, 4, 23, 2, 2, 2, 2)


class Res16UNet14A(Res16UNet14):
  PLANES = (32, 64, 128, 256, 128, 128, 96, 96)


class Res16UNet14A2(Res16UNet14A):
  LAYERS = (1, 1, 1, 1, 2, 2, 2, 2)


class Res16UNet14B(Res16UNet14):
  PLANES = (32, 64, 128, 256, 128, 128, 128, 128)


class Res16UNet14B2(Res16UNet14B):
  LAYERS = (1, 1, 1, 1, 2, 2, 2, 2)


class Res16UNet14B3(Res16UNet14B):
  LAYERS = (2, 2, 2, 2, 1, 1, 1, 1)


class Res16UNet14C(Res16UNet14):
  PLANES = (32, 64, 128, 256, 192, 192, 128, 128)


class Res16UNet14D(Res16UNet14):
  PLANES = (32, 64, 128, 256, 384, 384, 384, 384)


class Res16UNet18A(Res16UNet18):
  PLANES = (32, 64, 128, 256, 128, 128, 96, 96)


class Res16UNet18B(Res16UNet18):
  PLANES = (32, 64, 128, 256, 128, 128, 128, 128)


class Res16UNet18D(Res16UNet18):
  PLANES = (32, 64, 128, 256, 384, 384, 384, 384)


class Res16UNet34A(Res16UNet34):
  PLANES = (32, 64, 128, 256, 256, 128, 64, 64)


class Res16UNet34B(Res16UNet34):
  PLANES = (32, 64, 128, 256, 256, 128, 64, 32)


class Res16UNet34C(Res16UNet34):
  PLANES = (32, 64, 128, 256, 256, 128, 96, 96)

class MonitorRes16UNet34C(MonitorRes16UNet34):
  PLANES = (32, 64, 128, 256, 256, 128, 96, 96)

# Experimentally, worse than others
class Res16UNetLN14(Res16UNet14):
  NORM_TYPE = NormType.SPARSE_LAYER_NORM
  BLOCK = BasicBlockLN


class Res16UNetTemporalBase(Res16UNetBase):
  """
  Res16UNet that can take 4D independently. No temporal convolution.
  """
  CONV_TYPE = ConvType.SPATIAL_HYPERCUBE

  def __init__(self, in_channels, out_channels, config, D=4, **kwargs):
    super(Res16UNetTemporalBase, self).__init__(in_channels, out_channels, config, D, **kwargs)


class Res16UNetTemporal14(Res16UNet14, Res16UNetTemporalBase):
  pass


class Res16UNetTemporal18(Res16UNet18, Res16UNetTemporalBase):
  pass


class Res16UNetTemporal34(Res16UNet34, Res16UNetTemporalBase):
  pass


class Res16UNetTemporal50(Res16UNet50, Res16UNetTemporalBase):
  pass


class Res16UNetTemporal101(Res16UNet101, Res16UNetTemporalBase):
  pass


class Res16UNetTemporalIN14(Res16UNetTemporal14):
  NORM_TYPE = NormType.SPARSE_INSTANCE_NORM
  BLOCK = BasicBlockIN


class Res16UNetTemporalIN18(Res16UNetTemporal18):
  NORM_TYPE = NormType.SPARSE_INSTANCE_NORM
  BLOCK = BasicBlockIN


class Res16UNetTemporalIN34(Res16UNetTemporal34):
  NORM_TYPE = NormType.SPARSE_INSTANCE_NORM
  BLOCK = BasicBlockIN


class Res16UNetTemporalIN50(Res16UNetTemporal50):
  NORM_TYPE = NormType.SPARSE_INSTANCE_NORM
  BLOCK = BottleneckIN


class Res16UNetTemporalIN101(Res16UNetTemporal101):
  NORM_TYPE = NormType.SPARSE_INSTANCE_NORM
  BLOCK = BottleneckIN


class STRes16UNetBase(Res16UNetBase):

  CONV_TYPE = ConvType.SPATIAL_HYPERCUBE_TEMPORAL_HYPERCROSS

  def __init__(self, in_channels, out_channels, config, D=4, **kwargs):
    super(STRes16UNetBase, self).__init__(in_channels, out_channels, config, D, **kwargs)


class STRes16UNet14(STRes16UNetBase, Res16UNet14):
  pass


class STRes16UNet18(STRes16UNetBase, Res16UNet18):
  pass


class STRes16UNet34(STRes16UNetBase, Res16UNet34):
  pass


class STRes16UNet50(STRes16UNetBase, Res16UNet50):
  pass


class STRes16UNet101(STRes16UNetBase, Res16UNet101):
  pass


class STRes16UNet18A(STRes16UNet18):
  PLANES = (32, 64, 128, 256, 128, 128, 96, 96)


class STResTesseract16UNetBase(STRes16UNetBase):
  CONV_TYPE = ConvType.HYPERCUBE


class STResTesseract16UNet18A(STRes16UNet18A, STResTesseract16UNetBase):
  pass
