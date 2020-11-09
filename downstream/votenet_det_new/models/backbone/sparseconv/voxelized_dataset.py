# coding: utf-8
# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import sys
import numpy as np
import torch

from torch.utils.data import Dataset
from torch.utils.data._utils.collate import default_collate

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'utils'))

import MinkowskiEngine as ME

class VoxelizationDataset(Dataset):
    """
    Wrapper dataset which voxelize the original point clouds
    """
    def __init__(self, dataset, voxel_size=0.05):
        self.dataset = dataset
        self.VOXEL_SIZE = voxel_size

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        ret_dict = self.dataset[idx]
        
        # voxelization
        coords = np.floor(ret_dict['point_clouds'] / self.VOXEL_SIZE)
        inds = ME.utils.sparse_quantize(coords, return_index=True)
        coords = coords[inds].astype(np.int32)

        ret_dict['voxel'] = (coords, np.array(inds, dtype=np.int32))
        return ret_dict


def collate_fn(samples):
    data, voxel = [], []
    for sample in samples:
        data.append({w: sample[w] for w in sample if w != 'voxel'})
        voxel.append(sample['voxel'])

    # for non-voxel data, use default collate
    data_batch = default_collate(data)

    batch_ids = np.array(
        [b for b, v in enumerate(voxel) for _ in range(v[0].shape[0])])
    voxel_ids = np.concatenate([v[1] for v in voxel], 0)
    
    coords = np.concatenate([v[0] for v in voxel], 0)
    coords = np.concatenate([batch_ids[:, None], coords], 1)

    data_batch['voxel_coords'] = torch.from_numpy(coords)
    data_batch['voxel_inds'] = torch.from_numpy(voxel_ids)
    data_batch['voxel_feats'] = data_batch['point_clouds'].new_ones(batch_ids.shape[0], 3)

    return data_batch
