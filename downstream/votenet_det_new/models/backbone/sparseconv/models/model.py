# Copyright (c) Facebook, Inc. and its affiliates.
#  
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from MinkowskiEngine import MinkowskiNetwork


class Model(MinkowskiNetwork):
  """
  Base network for all sparse convnet

  By default, all networks are segmentation networks.
  """
  OUT_PIXEL_DIST = -1

  def __init__(self, in_channels, out_channels, config, D, **kwargs):
    super(Model, self).__init__(D)
    self.in_channels = in_channels
    self.out_channels = out_channels
    self.config = config


class HighDimensionalModel(Model):
  """
  Base network for all spatio (temporal) chromatic sparse convnet
  """

  def __init__(self, in_channels, out_channels, config, D, **kwargs):
    assert D > 4, "Num dimension smaller than 5"
    super(HighDimensionalModel, self).__init__(in_channels, out_channels, config, D, **kwargs)
