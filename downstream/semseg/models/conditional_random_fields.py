import torch
import torch.nn as nn
from torch.autograd import Variable

from MinkowskiEngine import SparseTensor, MinkowskiConvolution, MinkowskiConvolutionFunction, convert_to_int_tensor
from MinkowskiEngine import convert_region_type as me_convert_region_type

from models.model import HighDimensionalModel
from models.wrapper import Wrapper
from lib.math_functions import SparseMM
from models.modules.common import convert_region_type


class MeanField(HighDimensionalModel):
  """
  Abstract class for the bilateral and trilateral meanfield
  """
  OUT_PIXEL_DIST = 1

  # To use the model, must call initialize_coords before forward pass.
  # Once data is processed, call clear to reset the model before calling
  # initialize_coords
  def __init__(self, nchannels, spatial_sigma, chromatic_sigma, meanfield_iterations, is_temporal,
               config, **kwargs):
    D = 7 if is_temporal else 6
    self.is_temporal = is_temporal
    # Setup metadata
    super(MeanField, self).__init__(nchannels, nchannels, config, D=D)

    self.spatial_sigma = spatial_sigma
    self.chromatic_sigma = chromatic_sigma
    # temporal sigma is 1
    self.meanfield_iterations = meanfield_iterations

    self.pixel_dist = 1
    self.stride = 1
    self.dilation = 1

    conv = MinkowskiConvolution(
        nchannels,
        nchannels,
        kernel_size=config.wrapper_kernel_size,
        has_bias=False,
        region_type=convert_region_type(config.wrapper_region_type),
        dimension=D)

    # Create a region_offset
    self.region_type_, self.region_offset_, _ = me_convert_region_type(
        conv.region_type, 1, conv.kernel_size, conv.up_stride, conv.dilation, conv.region_offset,
        conv.axis_types, conv.dimension)

    # Check whether the mapping is required
    self.requires_mapping = False
    self.conv = conv
    self.kernel = conv.kernel
    self.convs = {}
    self.softmaxes = {}
    for i in range(self.meanfield_iterations):
      self.softmaxes[i] = nn.Softmax(dim=1)
      self.convs[i] = MinkowskiConvolutionFunction()

  def initialize_coords(self, model, in_coords, in_color):
    if torch.prod(convert_to_int_tensor(model.OUT_PIXEL_DIST, model.D)) != 1:
      self.requires_mapping = True

      out_coords = model.get_coords(model.OUT_PIXEL_DIST)
      out_color = model.permute_feature(in_color, model.OUT_PIXEL_DIST).int()

      # Tri/Bi-lateral grid
      out_tri_coords = torch.cat(
          [
              (torch.floor(out_coords[:, :3].float() / self.spatial_sigma)).int(),
              (torch.floor(out_color.float() / self.chromatic_sigma)).int(),
              out_coords[:, 3:]  # (time and) batch
          ],
          dim=1)
      orig_tri_coords = torch.cat(
          [
              (torch.floor(in_coords[:, :3].float() / self.spatial_sigma)).int(),
              (torch.floor(in_color.float() / self.chromatic_sigma)).int(),
              in_coords[:, 3:]  # (time and) batch
          ],
          dim=1)

      crf_tri_coords = torch.cat((out_tri_coords, orig_tri_coords), dim=0)

      # Create a trilateral Grid
      # super(MeanField, self).initialize_coords_with_duplicates(crf_tri_coords)

      # Create Sparse matrix mappings to/from the CRF coords
      in_cols = self.get_index_map(out_tri_coords, 1)
      self.in_mapping = torch.sparse.FloatTensor(
          torch.stack((in_cols.long(), torch.arange(in_cols.size(0), out=torch.LongTensor()))),
          torch.ones(in_cols.size(0)), torch.Size((self.n_rows, in_cols.size(0))))

      out_cols = self.get_index_map(orig_tri_coords, 1)
      self.out_mapping = torch.sparse.FloatTensor(
          torch.stack((torch.arange(out_cols.size(0), out=torch.LongTensor()), out_cols.long())),
          torch.ones(out_cols.size(0)), torch.Size((out_cols.size(0), self.n_rows)))

      if self.config.is_cuda:
        self.in_mapping, self.out_mapping = self.in_mapping.cuda(), self.out_mapping.cuda()

    else:
      self.requires_mapping = False

      out_coords = in_coords
      out_color = in_color
      crf_tri_coords = torch.cat(
          [
              (torch.floor(in_coords[:, :3].float() / self.spatial_sigma)).int(),
              (torch.floor(in_color.float() / self.chromatic_sigma)).int(),
              in_coords[:, 3:],  # (time and) batch
          ],
          dim=1)

    return crf_tri_coords

  def forward(self, x):
    xf = x.F
    if self.requires_mapping:
      # Map the network output to CRF input
      xf = SparseMM()(Variable(self.in_mapping), xf)

    out = xf
    for i in range(self.meanfield_iterations):  # Meanfield iteration
      # Normalization
      out = self.softmaxes[i](out)
      # Pairwise potential
      out = self.convs[i].apply(out, self.conv.kernel, x.pixel_dist, self.conv.stride,
                                self.conv.kernel_size, self.conv.dilation, self.region_type_,
                                self.region_offset_, x.coords_key, x.coords_key, x.coords_man)
      # Add unary
      out += xf

    if self.requires_mapping:
      # Map the CRF output to the origianl space
      out = SparseMM()(Variable(self.out_mapping), out)

    return SparseTensor(out, coords_key=x.coords_key, coords_manager=x.coords_man)


class BilateralCRF(Wrapper):
  OUT_PIXEL_DIST = 1

  def initialize_filter(self, NetClass, in_nchannel, out_nchannel, config):
    self.model = NetClass(in_nchannel, out_nchannel, config)
    self.filter = MeanField(
        out_nchannel,
        spatial_sigma=config.crf_spatial_sigma,
        chromatic_sigma=config.crf_chromatic_sigma,
        meanfield_iterations=config.meanfield_iterations,
        is_temporal=False,
        config=config)


class TrilateralCRF(Wrapper):
  OUT_PIXEL_DIST = 1

  def initialize_filter(self, NetClass, in_nchannel, out_nchannel, config):
    self.model = NetClass(in_nchannel, out_nchannel, config)
    self.filter = MeanField(
        out_nchannel,
        spatial_sigma=config.crf_spatial_sigma,
        chromatic_sigma=config.crf_chromatic_sigma,
        meanfield_iterations=config.meanfield_iterations,
        is_temporal=True,
        config=config)
