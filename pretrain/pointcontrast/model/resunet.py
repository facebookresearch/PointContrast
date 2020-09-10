from model.resnet import ResNetBase, get_norm
from model.modules.common import ConvType, NormType, conv, conv_tr
from model.modules.resnet_block import BasicBlock, Bottleneck, BasicBlockSN, BottleneckSN, BasicBlockIN, BottleneckIN, BasicBlockLN

from MinkowskiEngine import MinkowskiReLU
import MinkowskiEngine.MinkowskiOps as me


class ResUNetBase(ResNetBase):
  BLOCK = None
  PLANES = (64, 128, 256, 512, 256, 128, 128)
  DILATIONS = (1, 1, 1, 1, 1, 1)
  LAYERS = (2, 2, 2, 2, 2, 2)
  INIT_DIM = 64
  OUT_PIXEL_DIST = 1
  NORM_TYPE = NormType.BATCH_NORM
  NON_BLOCK_CONV_TYPE = ConvType.SPATIAL_HYPERCUBE
  CONV_TYPE = ConvType.SPATIAL_HYPERCUBE_TEMPORAL_HYPERCROSS

  # To use the model, must call initialize_coords before forward pass.
  # Once data is processed, call clear to reset the model before calling initialize_coords
  def __init__(self, in_channels, out_channels, config, D=3, **kwargs):
    super(ResUNetBase, self).__init__(in_channels, out_channels, config, D)

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
    self.conv1p1s1 = conv(
        in_channels,
        self.inplanes,
        kernel_size=space_n_time_m(config.conv1_kernel_size, 1),
        stride=1,
        dilation=1,
        conv_type=self.NON_BLOCK_CONV_TYPE,
        D=D)

    self.bn1 = get_norm(self.NORM_TYPE, self.PLANES[0], D, bn_momentum=bn_momentum)
    self.block1 = self._make_layer(
        self.BLOCK,
        self.PLANES[0],
        self.LAYERS[0],
        dilation=dilations[0],
        norm_type=self.NORM_TYPE,
        bn_momentum=bn_momentum)

    self.conv2p1s2 = conv(
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

    self.conv3p2s2 = conv(
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

    self.conv4p4s2 = conv(
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
    self.convtr4p8s2 = conv_tr(
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
    self.convtr5p4s2 = conv_tr(
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
    self.convtr6p2s2 = conv_tr(
        self.inplanes,
        self.PLANES[6],
        kernel_size=space_n_time_m(2, 1),
        upsample_stride=space_n_time_m(2, 1),
        dilation=1,
        bias=False,
        conv_type=self.NON_BLOCK_CONV_TYPE,
        D=D)
    self.bntr6 = get_norm(self.NORM_TYPE, self.PLANES[6], D, bn_momentum=bn_momentum)
    self.relu = MinkowskiReLU(inplace=True)

    self.final = conv(
        self.PLANES[6] + self.PLANES[0] * self.BLOCK.expansion,
        out_channels,
        kernel_size=1,
        stride=1,
        dilation=1,
        bias=True,
        D=D)

  def forward(self, x):
    out = self.conv1p1s1(x)
    out = self.bn1(out)
    out = self.relu(out)

    out_b1p1 = self.block1(out)

    out = self.conv2p1s2(out_b1p1)
    out = self.bn2(out)
    out = self.relu(out)

    out_b2p2 = self.block2(out)

    out = self.conv3p2s2(out_b2p2)
    out = self.bn3(out)
    out = self.relu(out)

    out_b3p4 = self.block3(out)

    out = self.conv4p4s2(out_b3p4)
    out = self.bn4(out)
    out = self.relu(out)

    # pixel_dist=8
    out = self.block4(out)

    out = self.convtr4p8s2(out)
    out = self.bntr4(out)
    out = self.relu(out)

    out = me.cat(out, out_b3p4)
    out = self.block5(out)

    out = self.convtr5p4s2(out)
    out = self.bntr5(out)
    out = self.relu(out)

    out = me.cat(out, out_b2p2)
    out = self.block6(out)

    out = self.convtr6p2s2(out)
    out = self.bntr6(out)
    out = self.relu(out)

    out = me.cat(out, out_b1p1)
    return self.final(out)


class ResUNet14(ResUNetBase):
  BLOCK = BasicBlock
  LAYERS = (1, 1, 1, 1, 1, 1)


class ResUNet18(ResUNetBase):
  BLOCK = BasicBlock
  LAYERS = (2, 2, 2, 2, 2, 2)


class ResUNet34(ResUNetBase):
  BLOCK = BasicBlock
  LAYERS = (3, 4, 6, 3, 2, 2)


class ResUNet50(ResUNetBase):
  BLOCK = Bottleneck
  LAYERS = (3, 4, 6, 3, 2, 2)


class ResUNet101(ResUNetBase):
  BLOCK = Bottleneck
  LAYERS = (3, 4, 23, 3, 2, 2)


class ResUNet14D(ResUNet14):
  PLANES = (64, 128, 256, 512, 512, 512, 512)


class ResUNet18D(ResUNet18):
  PLANES = (64, 128, 256, 512, 512, 512, 512)


class ResUNet34D(ResUNet34):
  PLANES = (64, 128, 256, 512, 512, 512, 512)


class ResUNet34E(ResUNet34):
  INIT_DIM = 32
  PLANES = (32, 64, 128, 256, 128, 64, 64)


class ResUNet34F(ResUNet34):
  INIT_DIM = 32
  PLANES = (32, 64, 128, 256, 128, 64, 32)


class ResUNetSN14(ResUNet14):
  NORM_TYPE = NormType.SPARSE_SWITCH_NORM
  BLOCK = BasicBlockSN


class ResUNetSN18(ResUNet18):
  NORM_TYPE = NormType.SPARSE_SWITCH_NORM
  BLOCK = BasicBlockSN


class ResUNetSN34(ResUNet34):
  NORM_TYPE = NormType.SPARSE_SWITCH_NORM
  BLOCK = BasicBlockSN


class ResUNetSN50(ResUNet50):
  NORM_TYPE = NormType.SPARSE_SWITCH_NORM
  BLOCK = BottleneckSN


class ResUNetSN101(ResUNet101):
  NORM_TYPE = NormType.SPARSE_SWITCH_NORM
  BLOCK = BottleneckSN


class ResUNetIN14(ResUNet14):
  NORM_TYPE = NormType.SPARSE_INSTANCE_NORM
  BLOCK = BasicBlockIN


class ResUNetIN18(ResUNet18):
  NORM_TYPE = NormType.SPARSE_INSTANCE_NORM
  BLOCK = BasicBlockIN


class ResUNetIN34(ResUNet34):
  NORM_TYPE = NormType.SPARSE_INSTANCE_NORM
  BLOCK = BasicBlockIN


class ResUNetIN34E(ResUNet34E):
  NORM_TYPE = NormType.SPARSE_INSTANCE_NORM
  BLOCK = BasicBlockIN


class ResUNetIN50(ResUNet50):
  NORM_TYPE = NormType.SPARSE_INSTANCE_NORM
  BLOCK = BottleneckIN


class ResUNetIN101(ResUNet101):
  NORM_TYPE = NormType.SPARSE_INSTANCE_NORM
  BLOCK = BottleneckIN


# Experimentally, worse than others
class ResUNetLN14(ResUNet14):
  NORM_TYPE = NormType.SPARSE_LAYER_NORM
  BLOCK = BasicBlockLN


class ResUNetTemporalBase(ResUNetBase):
  """
  ResUNet that can take 4D independently. No temporal convolution.
  """
  CONV_TYPE = ConvType.SPATIAL_HYPERCUBE

  def __init__(self, in_channels, out_channels, config, D=4, **kwargs):
    super(ResUNetTemporalBase, self).__init__(in_channels, out_channels, config, D, **kwargs)


class ResUNetTemporal14(ResUNet14, ResUNetTemporalBase):
  pass


class ResUNetTemporal18(ResUNet18, ResUNetTemporalBase):
  pass


class ResUNetTemporal34(ResUNet34, ResUNetTemporalBase):
  pass


class ResUNetTemporal50(ResUNet50, ResUNetTemporalBase):
  pass


class ResUNetTemporal101(ResUNet101, ResUNetTemporalBase):
  pass


class ResUNetTemporalIN14(ResUNetTemporal14):
  NORM_TYPE = NormType.SPARSE_INSTANCE_NORM
  BLOCK = BasicBlockIN


class ResUNetTemporalIN18(ResUNetTemporal18):
  NORM_TYPE = NormType.SPARSE_INSTANCE_NORM
  BLOCK = BasicBlockIN


class ResUNetTemporalIN34(ResUNetTemporal34):
  NORM_TYPE = NormType.SPARSE_INSTANCE_NORM
  BLOCK = BasicBlockIN


class ResUNetTemporalIN50(ResUNetTemporal50):
  NORM_TYPE = NormType.SPARSE_INSTANCE_NORM
  BLOCK = BottleneckIN


class ResUNetTemporalIN101(ResUNetTemporal101):
  NORM_TYPE = NormType.SPARSE_INSTANCE_NORM
  BLOCK = BottleneckIN


class STResUNetBase(ResUNetBase):

  CONV_TYPE = ConvType.SPATIAL_HYPERCUBE_TEMPORAL_HYPERCROSS

  def __init__(self, in_channels, out_channels, config, D=4, **kwargs):
    super(STResUNetBase, self).__init__(in_channels, out_channels, config, D, **kwargs)


class STResUNet14(STResUNetBase, ResUNet14):
  pass


class STResUNet18(STResUNetBase, ResUNet18):
  pass


class STResUNet34(STResUNetBase, ResUNet34):
  pass


class STResUNet50(STResUNetBase, ResUNet50):
  pass


class STResUNet101(STResUNetBase, ResUNet101):
  pass


class STResUNetIN14(STResUNet14):
  NORM_TYPE = NormType.SPARSE_INSTANCE_NORM
  BLOCK = BasicBlockIN


class STResUNetIN18(STResUNet18):
  NORM_TYPE = NormType.SPARSE_INSTANCE_NORM
  BLOCK = BasicBlockIN


class STResUNetIN34(STResUNet34):
  NORM_TYPE = NormType.SPARSE_INSTANCE_NORM
  BLOCK = BasicBlockIN


class STResUNetIN50(STResUNet50):
  NORM_TYPE = NormType.SPARSE_INSTANCE_NORM
  BLOCK = BottleneckIN


class STResUNetIN101(STResUNet101):
  NORM_TYPE = NormType.SPARSE_INSTANCE_NORM
  BLOCK = BottleneckIN


class STResTesseractUNetBase(STResUNetBase):
  CONV_TYPE = ConvType.HYPERCUBE


class STResTesseractUNet14(STResTesseractUNetBase, ResUNet14):
  pass


class STResTesseractUNet18(STResTesseractUNetBase, ResUNet18):
  pass


class STResTesseractUNet34(STResTesseractUNetBase, ResUNet34):
  pass


class STResTesseractUNet50(STResTesseractUNetBase, ResUNet50):
  pass


class STResTesseractUNet101(STResTesseractUNetBase, ResUNet101):
  pass
