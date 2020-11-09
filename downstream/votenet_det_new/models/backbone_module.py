# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
import os

from models.backbone.pointnet2.pointnet2_modules import PointnetSAModuleVotes, PointnetFPModule
from models.backbone.pointnet2.pointnet2_utils import furthest_point_sample

import MinkowskiEngine as ME


class Pointnet2Backbone(nn.Module):
    r"""
       Backbone network for point cloud feature learning.
       Based on Pointnet++ single-scale grouping network. 
        
       Parameters
       ----------
       input_feature_dim: int
            Number of input channels in the feature descriptor for each point.
            e.g. 3 for RGB.
    """
    def __init__(self, input_feature_dim=0):
        super().__init__()

        self.sa1 = PointnetSAModuleVotes(
                npoint=2048,
                radius=0.2,
                nsample=64,
                mlp=[input_feature_dim, 64, 64, 128],
                use_xyz=True,
                normalize_xyz=True
            )

        self.sa2 = PointnetSAModuleVotes(
                npoint=1024,
                radius=0.4,
                nsample=32,
                mlp=[128, 128, 128, 256],
                use_xyz=True,
                normalize_xyz=True
            )

        self.sa3 = PointnetSAModuleVotes(
                npoint=512,
                radius=0.8,
                nsample=16,
                mlp=[256, 128, 128, 256],
                use_xyz=True,
                normalize_xyz=True
            )

        self.sa4 = PointnetSAModuleVotes(
                npoint=256,
                radius=1.2,
                nsample=16,
                mlp=[256, 128, 128, 256],
                use_xyz=True,
                normalize_xyz=True
            )

        self.fp1 = PointnetFPModule(mlp=[256+256,256,256])
        self.fp2 = PointnetFPModule(mlp=[256+256,256,256])

    def _break_up_pc(self, pc):
        xyz = pc[..., 0:3].contiguous()
        features = (
            pc[..., 3:].transpose(1, 2).contiguous()
            if pc.size(-1) > 3 else None
        )

        return xyz, features

    def forward(self, pointcloud: torch.cuda.FloatTensor, end_points=None):
        r"""
            Forward pass of the network

            Parameters
            ----------
            pointcloud: Variable(torch.cuda.FloatTensor)
                (B, N, 3 + input_feature_dim) tensor
                Point cloud to run predicts on
                Each point in the point-cloud MUST
                be formated as (x, y, z, features...)

            Returns
            ----------
            end_points: {XXX_xyz, XXX_features, XXX_inds}
                XXX_xyz: float32 Tensor of shape (B,K,3)
                XXX_features: float32 Tensor of shape (B,K,D)
                XXX-inds: int64 Tensor of shape (B,K) values in [0,N-1]
        """
        if not end_points: end_points = {}
        batch_size = pointcloud.shape[0]

        xyz, features = self._break_up_pc(pointcloud)

        # --------- 4 SET ABSTRACTION LAYERS ---------
        xyz, features, fps_inds = self.sa1(xyz, features)
        end_points['sa1_inds'] = fps_inds
        end_points['sa1_xyz'] = xyz
        end_points['sa1_features'] = features

        xyz, features, fps_inds = self.sa2(xyz, features) # this fps_inds is just 0,1,...,1023
        end_points['sa2_inds'] = fps_inds
        end_points['sa2_xyz'] = xyz
        end_points['sa2_features'] = features

        xyz, features, fps_inds = self.sa3(xyz, features) # this fps_inds is just 0,1,...,511
        end_points['sa3_xyz'] = xyz
        end_points['sa3_features'] = features

        xyz, features, fps_inds = self.sa4(xyz, features) # this fps_inds is just 0,1,...,255
        end_points['sa4_xyz'] = xyz
        end_points['sa4_features'] = features

        # --------- 2 FEATURE UPSAMPLING LAYERS --------
        features = self.fp1(end_points['sa3_xyz'], end_points['sa4_xyz'], end_points['sa3_features'], end_points['sa4_features'])
        features = self.fp2(end_points['sa2_xyz'], end_points['sa3_xyz'], end_points['sa2_features'], features)
        end_points['fp2_features'] = features
        end_points['fp2_xyz'] = end_points['sa2_xyz']
        num_seed = end_points['fp2_xyz'].shape[1]
        end_points['fp2_inds'] = end_points['sa1_inds'][:,0:num_seed] # indices among the entire input point clouds
        return end_points


class SparseConvBackbone(nn.Module):

    def __init__(self, 
        input_feature_dim=3,
        output_feature_dim=256,
        num_seed=1024, 
        model='Res16UNet34C',
        config=None):
    
        super().__init__()
        from models.backbone.sparseconv.config import get_config
        from models.backbone.sparseconv.models import load_model

        config = get_config(["--conv1_kernel_size", "3", "--model", model])

        # from pdb import set_trace; set_trace()
        self.net = load_model(model)(
            input_feature_dim, output_feature_dim, config)
        self.num_seed = num_seed

    def forward(self, points, coords, feats, inds, end_points=None):
        inputs = ME.SparseTensor(feats.cpu(), coords=coords.cpu().int()).to(coords.device)
        outputs = self.net(inputs)
        features = outputs.F

        # randomly down-sample to num_seed points & create batches
        bsz, num_points, _ = points.size()
        points = points.view(-1, 3)
        batch_ids = coords[:, 0]
        voxel_ids = inds + batch_ids * num_points

        sampled_inds, sampled_feartures, sampled_points = [], [], []
        for b in range(bsz):
            sampled_id = furthest_point_sample(
                points[voxel_ids[batch_ids == b]].unsqueeze(0), 
                self.num_seed).squeeze(0).long()
            
            sampled_inds.append(inds[batch_ids == b][sampled_id])
            sampled_feartures.append(features[batch_ids == b][sampled_id])
            sampled_points.append(points[voxel_ids[batch_ids == b]][sampled_id])

        end_points['fp2_features'] = torch.stack(sampled_feartures, 0).transpose(1, 2)
        end_points['fp2_xyz'] = torch.stack(sampled_points, 0)
        end_points['fp2_inds'] = torch.stack(sampled_inds, 0)
        
        # from pdb import set_trace; set_trace()
        return end_points

if __name__=='__main__':
    backbone_net = Pointnet2Backbone(input_feature_dim=3).cuda()
    print(backbone_net)
    backbone_net.eval()
    out = backbone_net(torch.rand(16,20000,6).cuda())
    for key in sorted(out.keys()):
        print(key, '\t', out[key].shape)
