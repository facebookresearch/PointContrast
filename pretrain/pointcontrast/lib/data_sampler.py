import torch
from torch.utils.data.sampler import Sampler
import torch.distributed as dist

import math

class InfSampler(Sampler):
  def __init__(self, data_source, shuffle=False):
    self.data_source = data_source
    self.shuffle = shuffle
    self.reset_permutation()

  def reset_permutation(self):
    perm = len(self.data_source)
    if self.shuffle:
      perm = torch.randperm(perm)
    self._perm = perm.tolist()

  def __iter__(self):
    return self

  def __next__(self):
    if len(self._perm) == 0:
      self.reset_permutation()
    return self._perm.pop()

  def __len__(self):
    return len(self.data_source)

  next = __next__  # Python 2 compatibility


class DistributedInfSampler(InfSampler):
  def __init__(self, data_source, num_replicas=None, rank=None, shuffle=True):
    if num_replicas is None:
      if not dist.is_available():
          raise RuntimeError("Requires distributed package to be available")
      num_replicas = dist.get_world_size()
    if rank is None:
      if not dist.is_available():
          raise RuntimeError("Requires distributed package to be available")
      rank = dist.get_rank()

    self.data_source = data_source
    self.num_replicas = num_replicas
    self.rank = rank
    self.epoch = 0
    self.it = 0
    self.num_samples = int(math.ceil(len(self.data_source) * 1.0 / self.num_replicas))
    self.total_size = self.num_samples * self.num_replicas
    self.shuffle = shuffle
    self.reset_permutation()
  
  def __next__(self):
    it = self.it * self.num_replicas + self.rank
    value = self._perm[it % len(self._perm)]
    self.it = self.it + 1
    
    if (self.it * self.num_replicas) >= len(self._perm):
      self.reset_permutation()
      self.it = 0
    return value

  def __len__(self):
    return self.num_samples
