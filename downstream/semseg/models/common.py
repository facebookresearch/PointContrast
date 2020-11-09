# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import MinkowskiEngine as ME


def get_norm(norm_type, num_feats, bn_momentum=0.05, D=-1):
  if norm_type == 'BN':
    return ME.MinkowskiBatchNorm(num_feats, momentum=bn_momentum)
  elif norm_type == 'IN':
    return ME.MinkowskiInstanceNorm(num_feats, dimension=D)
  else:
    raise ValueError(f'Type {norm_type}, not defined')
