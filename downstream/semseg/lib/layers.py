import torch
import torch.nn as nn

from MinkowskiEngine import MinkowskiGlobalPooling, MinkowskiBroadcastAddition, MinkowskiBroadcastMultiplication


class MinkowskiLayerNorm(nn.Module):

  def __init__(self, num_features, eps=1e-5, D=-1):
    super(MinkowskiLayerNorm, self).__init__()
    self.num_features = num_features
    self.eps = eps
    self.weight = nn.Parameter(torch.ones(1, num_features))
    self.bias = nn.Parameter(torch.zeros(1, num_features))

    self.mean_in = MinkowskiGlobalPooling(dimension=D)
    self.glob_sum = MinkowskiBroadcastAddition(dimension=D)
    self.glob_sum2 = MinkowskiBroadcastAddition(dimension=D)
    self.glob_mean = MinkowskiGlobalPooling(dimension=D)
    self.glob_times = MinkowskiBroadcastMultiplication(dimension=D)
    self.D = D
    self.reset_parameters()

  def __repr__(self):
    s = f'(D={self.D})'
    return self.__class__.__name__ + s

  def reset_parameters(self):
    self.weight.data.fill_(1)
    self.bias.data.zero_()

  def _check_input_dim(self, input):
    if input.F.dim() != 2:
      raise ValueError('expected 2D input (got {}D input)'.format(input.dim()))

  def forward(self, x):
    self._check_input_dim(x)
    mean = self.mean_in(x).F.mean(-1, keepdim=True)
    mean = mean + torch.zeros(mean.size(0), self.num_features).type_as(mean)
    temp = self.glob_sum(x.F, -mean)**2
    var = self.glob_mean(temp.data).mean(-1, keepdim=True)
    var = var + torch.zeros(var.size(0), self.num_features).type_as(var)
    instd = 1 / (var + self.eps).sqrt()

    x = self.glob_times(self.glob_sum2(x, -mean), instd)
    return x * self.weight + self.bias


class MinkowskiInstanceNorm(nn.Module):

  def __init__(self, num_features, eps=1e-5, D=-1):
    super(MinkowskiInstanceNorm, self).__init__()
    self.eps = eps
    self.weight = nn.Parameter(torch.ones(1, num_features))
    self.bias = nn.Parameter(torch.zeros(1, num_features))

    self.mean_in = MinkowskiGlobalPooling(dimension=D)
    self.glob_sum = MinkowskiBroadcastAddition(dimension=D)
    self.glob_sum2 = MinkowskiBroadcastAddition(dimension=D)
    self.glob_mean = MinkowskiGlobalPooling(dimension=D)
    self.glob_times = MinkowskiBroadcastMultiplication(dimension=D)
    self.D = D
    self.reset_parameters()

  def __repr__(self):
    s = f'(pixel_dist={self.pixel_dist}, D={self.D})'
    return self.__class__.__name__ + s

  def reset_parameters(self):
    self.weight.data.fill_(1)
    self.bias.data.zero_()

  def _check_input_dim(self, input):
    if input.dim() != 2:
      raise ValueError('expected 2D input (got {}D input)'.format(input.dim()))

  def forward(self, x):
    self._check_input_dim(x)
    mean_in = self.mean_in(x)
    temp = self.glob_sum(x, -mean_in)**2
    var_in = self.glob_mean(temp.data)
    instd_in = 1 / (var_in + self.eps).sqrt()

    x = self.glob_times(self.glob_sum2(x, -mean_in), instd_in)
    return x * self.weight + self.bias
