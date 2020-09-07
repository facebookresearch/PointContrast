# Written by Chris Choy <chrischoy@ai.stanford.edu>
# Distributed under MIT License
import logging
import random
import torch
import torch.utils.data
import numpy as np
import glob
import os
import copy
from tqdm import tqdm
from scipy.linalg import expm, norm

from util.pointcloud import get_matching_indices, get_self_matching_indices, make_open3d_point_cloud
import lib.transforms as t

import MinkowskiEngine as ME

import open3d as o3d

kitti_cache = {}
kitti_icp_cache = {}


def default_collate_pair_fn(list_data):
  xyz0, xyz1, coords0, coords1, feats0, feats1, matching_inds, trans = list(
      zip(*list_data))
  xyz_batch0, coords_batch0, feats_batch0 = [], [], []
  xyz_batch1, coords_batch1, feats_batch1 = [], [], []
  matching_inds_batch, trans_batch, len_batch = [], [], []

  batch_id = 0
  curr_start_inds = np.zeros((1, 2))
  for batch_id, _ in enumerate(coords0):

    N0 = coords0[batch_id].shape[0]
    N1 = coords1[batch_id].shape[0]

    xyz_batch0.append(torch.from_numpy(xyz0[batch_id]))
    coords_batch0.append(
        torch.cat((torch.from_numpy(
            coords0[batch_id]).int(), torch.ones(N0, 1).int() * batch_id), 1))
    feats_batch0.append(torch.from_numpy(feats0[batch_id]))

    xyz_batch1.append(torch.from_numpy(xyz1[batch_id]))
    coords_batch1.append(
        torch.cat((torch.from_numpy(
            coords1[batch_id]).int(), torch.ones(N1, 1).int() * batch_id), 1))
    feats_batch1.append(torch.from_numpy(feats1[batch_id]))

    trans_batch.append(torch.from_numpy(trans[batch_id]))
    matching_inds_batch.append(
        torch.from_numpy(np.array(matching_inds[batch_id]) + curr_start_inds))
    len_batch.append([N0, N1])

    # Move the head
    curr_start_inds[0, 0] += N0
    curr_start_inds[0, 1] += N1

  # Concatenate all lists
  xyz_batch0 = torch.cat(xyz_batch0, 0).float()
  coords_batch0 = torch.cat(coords_batch0, 0).int()
  feats_batch0 = torch.cat(feats_batch0, 0).float()
  xyz_batch1 = torch.cat(xyz_batch1, 0).float()
  coords_batch1 = torch.cat(coords_batch1, 0).int()
  feats_batch1 = torch.cat(feats_batch1, 0).float()
  trans_batch = torch.cat(trans_batch, 0).float()
  matching_inds_batch = torch.cat(matching_inds_batch, 0).int()
  return {
      'pcd0': xyz_batch0,
      'pcd1': xyz_batch1,
      'sinput0_C': coords_batch0,
      'sinput0_F': feats_batch0,
      'sinput1_C': coords_batch1,
      'sinput1_F': feats_batch1,
      'correspondences': matching_inds_batch,
      'T_gt': trans_batch,
      'len_batch': len_batch,
  }

def shapenet_collate_pair_fn(list_data):
  xyz0, xyz1, coords0, coords1, feats0, feats1, matching_inds, trans, labels, xyz_orig, coords_orig, feats_orig = list(
      zip(*list_data))
  xyz_batch0, coords_batch0, feats_batch0 = [], [], []
  xyz_batch1, coords_batch1, feats_batch1 = [], [], []
  xyz_batch_orig, coords_batch_orig, feats_batch_orig = [], [], []
  matching_inds_batch, trans_batch, len_batch = [], [], []
  labels_batch = []

  batch_id = 0
  curr_start_inds = np.zeros((1, 2))
  for batch_id, _ in enumerate(coords0):

    N_orig = coords_orig[batch_id].shape[0]
    N0 = coords0[batch_id].shape[0]
    N1 = coords1[batch_id].shape[0]

    xyz_batch_orig.append(torch.from_numpy(xyz_orig[batch_id]))
    coords_batch_orig.append(
        torch.cat((torch.from_numpy(
            coords_orig[batch_id]).int(), torch.ones(N_orig, 1).int() * batch_id), 1))
    feats_batch_orig.append(torch.from_numpy(feats_orig[batch_id]))

    xyz_batch0.append(torch.from_numpy(xyz0[batch_id]))
    coords_batch0.append(
        torch.cat((torch.from_numpy(
            coords0[batch_id]).int(), torch.ones(N0, 1).int() * batch_id), 1))
    feats_batch0.append(torch.from_numpy(feats0[batch_id]))

    xyz_batch1.append(torch.from_numpy(xyz1[batch_id]))
    coords_batch1.append(
        torch.cat((torch.from_numpy(
            coords1[batch_id]).int(), torch.ones(N1, 1).int() * batch_id), 1))
    feats_batch1.append(torch.from_numpy(feats1[batch_id]))

    trans_batch.append(torch.from_numpy(trans[batch_id]))
    # TODO(s9xie): what causes the crash?
    if len(matching_inds[batch_id]) == 0:
      matching_inds[batch_id].extend([0, 0])
    matching_inds_batch.append(
        torch.from_numpy(np.array(matching_inds[batch_id]) + curr_start_inds))
    len_batch.append([N0, N1])
    # print(labels)
    labels_batch.append(labels[batch_id])

    # Move the head
    curr_start_inds[0, 0] += N0
    curr_start_inds[0, 1] += N1

  # Concatenate all lists
  xyz_batch0 = torch.cat(xyz_batch0, 0).float()
  coords_batch0 = torch.cat(coords_batch0, 0).int()
  feats_batch0 = torch.cat(feats_batch0, 0).float()
  xyz_batch1 = torch.cat(xyz_batch1, 0).float()
  coords_batch1 = torch.cat(coords_batch1, 0).int()
  feats_batch1 = torch.cat(feats_batch1, 0).float()
  trans_batch = torch.cat(trans_batch, 0).float()
  matching_inds_batch = torch.cat(matching_inds_batch, 0).int()
  labels_batch = torch.tensor(labels_batch)
  xyz_batch_orig = torch.cat(xyz_batch_orig, 0).float()
  coords_batch_orig = torch.cat(coords_batch_orig, 0).int()
  feats_batch_orig = torch.cat(feats_batch_orig, 0).float()
  return {
      'pcd0': xyz_batch0,
      'pcd1': xyz_batch1,
      'sinput0_C': coords_batch0,
      'sinput0_F': feats_batch0,
      'sinput1_C': coords_batch1,
      'sinput1_F': feats_batch1,
      'correspondences': matching_inds_batch,
      'T_gt': trans_batch,
      'len_batch': len_batch,
      'labels': labels_batch,
      'pcd_orig': xyz_batch_orig,
      'sinput_orig_C': coords_batch_orig,
      'sinput_orig_F': feats_batch_orig
  }


# Rotation matrix along axis with angle theta
def M(axis, theta):
  return expm(np.cross(np.eye(3), axis / norm(axis) * theta))


def sample_random_trans(pcd, randg, rotation_range=360):
  # first minus mean (centering), then rotation 360 degrees
  T = np.eye(4)
  R = M(randg.rand(3) - 0.5, rotation_range * np.pi / 180.0 * (randg.rand(1) - 0.5))
  T[:3, :3] = R
  T[:3, 3] = R.dot(-np.mean(pcd, axis=0))
  return T

def sample_dummy_trans(pcd):
  T = np.eye(4)
  R = np.eye(3)
  T[:3, 3] = R.dot(-np.mean(pcd, axis=0))
  return T

class PairDataset(torch.utils.data.Dataset):
  AUGMENT = None

  def __init__(self,
               phase,
               transform=None,
               random_rotation=True,
               random_scale=True,
               manual_seed=False,
               config=None):
    self.phase = phase
    self.files = []
    self.data_objects = []
    self.transform = transform
    self.voxel_size = config.voxel_size
    self.matching_search_voxel_size = \
        config.voxel_size * config.positive_pair_search_voxel_size_multiplier

    self.random_scale = random_scale
    self.min_scale = config.min_scale
    self.max_scale = config.max_scale
    self.random_rotation = random_rotation
    self.rotation_range = config.rotation_range
    self.randg = np.random.RandomState()
    if manual_seed:
      self.reset_seed()

  def reset_seed(self, seed=0):
    logging.info(f"Resetting the data loader seed to {seed}")
    self.randg.seed(seed)

  def apply_transform(self, pts, trans):
    R = trans[:3, :3]
    T = trans[:3, 3]
    pts = pts @ R.T + T
    return pts

  def __len__(self):
    return len(self.files)


class IndoorPairDataset(PairDataset):
  OVERLAP_RATIO = None
  AUGMENT = None

  def __init__(self,
               phase,
               transform=None,
               random_rotation=True,
               random_scale=True,
               manual_seed=False,
               config=None):
    PairDataset.__init__(self, phase, transform, random_rotation, random_scale,
                         manual_seed, config)
    self.root = root = config.threed_match_dir
    logging.info(f"IndoorPair Loading the subset {phase} from {root}")

    subset_names = open(self.DATA_FILES[phase]).read().split()
    for name in subset_names:
      fname = name + "*%.2f.txt" % self.OVERLAP_RATIO
      fnames_txt = glob.glob(root + "/" + fname)
      assert len(fnames_txt) > 0, f"Make sure that the path {root} has data {fname}"
      for fname_txt in fnames_txt:
        with open(fname_txt) as f:
          content = f.readlines()
        fnames = [x.strip().split() for x in content]
        for fname in fnames:
          self.files.append([fname[0], fname[1]])

  def __getitem__(self, idx):
    file0 = os.path.join(self.root, self.files[idx][0])
    file1 = os.path.join(self.root, self.files[idx][1])
    data0 = np.load(file0)
    data1 = np.load(file1)
    xyz0 = data0["pcd"]
    xyz1 = data1["pcd"]
    color0 = data0["color"]
    color1 = data1["color"]

    print("before_voxliaze: ", xyz0.shape)
    
    matching_search_voxel_size = self.matching_search_voxel_size

    if self.random_scale and random.random() < 0.95:
      scale = self.min_scale + \
          (self.max_scale - self.min_scale) * random.random()
      matching_search_voxel_size *= scale
      xyz0 = scale * xyz0
      xyz1 = scale * xyz1

    if self.random_rotation:
      T0 = sample_random_trans(xyz0, self.randg, self.rotation_range)
      T1 = sample_random_trans(xyz1, self.randg, self.rotation_range)
      trans = T1 @ np.linalg.inv(T0)

      xyz0 = self.apply_transform(xyz0, T0)
      xyz1 = self.apply_transform(xyz1, T1)
    else:
      trans = np.identity(4)
    # Voxelization
    sel0 = ME.utils.sparse_quantize(xyz0 / self.voxel_size, return_index=True)
    sel1 = ME.utils.sparse_quantize(xyz1 / self.voxel_size, return_index=True)

    # Make point clouds using voxelized points
    pcd0 = make_open3d_point_cloud(xyz0)
    pcd1 = make_open3d_point_cloud(xyz1)

    # Select features and points using the returned voxelized indices
    pcd0.colors = o3d.utility.Vector3dVector(color0[sel0])
    pcd1.colors = o3d.utility.Vector3dVector(color1[sel1])
    pcd0.points = o3d.utility.Vector3dVector(np.array(pcd0.points)[sel0])
    pcd1.points = o3d.utility.Vector3dVector(np.array(pcd1.points)[sel1])
    # Get matches
    matches = get_matching_indices(pcd0, pcd1, trans, matching_search_voxel_size)
    # Get features
    npts0 = len(pcd0.colors)
    npts1 = len(pcd1.colors)

    feats_train0, feats_train1 = [], []
    feats_train0.append(np.ones((npts0, 3)))
    feats_train1.append(np.ones((npts1, 3)))

    feats0 = np.hstack(feats_train0)
    feats1 = np.hstack(feats_train1)

    # Get coords
    xyz0 = np.array(pcd0.points)
    xyz1 = np.array(pcd1.points)

    coords0 = np.floor(xyz0 / self.voxel_size)
    coords1 = np.floor(xyz1 / self.voxel_size)
    print("after_voxliaze: ", coords0.shape)

    if self.transform:
      coords0, feats0 = self.transform(coords0, feats0)
      coords1, feats1 = self.transform(coords1, feats1)

    # NB(s9xie): xyz are coordinates in the original system; 
    # coords are sparse conv grid coords. (subject to a scaling factor)
    # coords0 -> sinput0_C
    # trans is T0*T1^-1
    # xyz0 (12419, 3)
    # coords0 (12419, 3)
    # feats0 (12419, 1)

    return (xyz0, xyz1, coords0, coords1, feats0, feats1, matches, trans)


class ThreeDMatchPairDataset(IndoorPairDataset):
  OVERLAP_RATIO = 0.3
  DATA_FILES = {
      'train': './config/train_3dmatch.txt',
      'val': './config/val_3dmatch.txt',
      'test': './config/test_3dmatch.txt'
  }

class SingleDataset(torch.utils.data.Dataset):
  AUGMENT = None

  def __init__(self,
               phase,
               transform=None,
               random_rotation=True,
               random_scale=True,
               manual_seed=False,
               config=None):
    self.phase = phase
    self.files = []
    self.data_objects = []
    self.transform = transform
    self.voxel_size = config.voxel_size
    self.matching_search_voxel_size = \
        config.voxel_size * config.positive_pair_search_voxel_size_multiplier

    self.random_scale = random_scale
    self.min_scale = config.min_scale
    self.max_scale = config.max_scale
    self.random_rotation = random_rotation
    self.rotation_range = config.rotation_range
    self.randg = np.random.RandomState()
    if manual_seed:
      self.reset_seed()

  def reset_seed(self, seed=0):
    logging.info(f"Resetting the data loader seed to {seed}")
    self.randg.seed(seed)

  def apply_transform(self, pts, trans):
    R = trans[:3, :3]
    T = trans[:3, 3]
    pts = pts @ R.T + T
    return pts

  def pc_normalize(self, pc):
    """ pc: NxC, return NxC """
    l = pc.shape[0]
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc

  def __len__(self):
    return len(self.files)


class ShapeNetDataset(SingleDataset):
  AUGMENT = None

  def __init__(self,
               phase,
               transform=None,
               random_rotation=True,
               random_scale=True,
               manual_seed=False,
               config=None):
    SingleDataset.__init__(self, phase, transform, random_rotation, random_scale,
                         manual_seed, config)
    self.root = root = config.shapenet_dir
    self.phase = phase
    self.npoints = config.shapenet_npoints
    logging.info(f"ShapeNet Loading the subset {phase} from {root}")
    self.normalize = True

    # NB(s9xie): train on trainval set and val on test set
    if phase == "train":
        self.train_files = np.load(os.path.join(self.root, "new_train_files.npy"), allow_pickle=True)
        self.val_files = np.load(os.path.join(self.root, "new_val_files.npy"), allow_pickle=True)
        self.files = np.concatenate([self.train_files, self.val_files])
    elif phase == "val":
        self.files = np.load(os.path.join(self.root, "new_test_files.npy"), allow_pickle=True)
    print(self.files[0])
    # TODO(s9xie): we don't need to load labels for now
    # self.labels = []
    # for data_path in tqdm(self.files, total=len(self.files), desc="dataset: retrieve all labels"):
    #     example = np.load(data_path, allow_pickle=True)
    #     self.labels.append(example["cid"])
    # self.label_dataset_size = len(self.files)
    
  def __getitem__(self, idx):
    data0 = np.load(self.files[idx], allow_pickle=True)
    label = data0["cid"]
    xyz_raw = data0["point_cloud"]
    if self.phase == "train":
      choice1 = np.random.choice(xyz_raw.shape[0], self.npoints, replace=False)
      choice2 = np.random.choice(xyz_raw.shape[0], self.npoints, replace=False)
      xyz0 = xyz_raw[choice1, :]
      xyz1 = xyz_raw[choice2, :]
    else:
      xyz0 = xyz_raw[:self.npoints, :]
      xyz1 = xyz_raw[:self.npoints, :]

    if self.normalize:
      xyz0 = self.pc_normalize(xyz0)
      xyz1 = self.pc_normalize(xyz1)
    

    # xyz1 = copy.deepcopy(xyz0)
    xyz_orig = copy.deepcopy(xyz0)

    #color0 = data0["color"]
    # print(xyz0.shape)
    color0 = np.ones([xyz0.shape[0], 3])
    color1 = copy.deepcopy(color0)
    color_orig = copy.deepcopy(color0)
    # print(color0.shape)
    T0 = sample_random_trans(xyz0, self.randg, self.rotation_range)
    T1 = sample_random_trans(xyz1, self.randg, self.rotation_range)
    T_dummy = sample_dummy_trans(xyz_orig)
    trans = T1 @ np.linalg.inv(T0)

    xyz0 = self.apply_transform(xyz0, T0)
    xyz1 = self.apply_transform(xyz1, T1)
    # print(xyz_orig.mean())
    xyz_orig = self.apply_transform(xyz_orig, T_dummy)
    # print(xyz_orig.mean())

    # if self.phase == "train":
    #   print(xyz0[0:5])
    #   print(xyz1[0:5])

    matching_search_voxel_size = self.matching_search_voxel_size
    if self.random_scale and random.random() < 0.95:
      scale = self.min_scale + \
          (self.max_scale - self.min_scale) * random.random()
      matching_search_voxel_size *= scale
      xyz0 = scale * xyz0
      xyz1 = scale * xyz1
      # xyz_orig = scale * xyz_orig

    # Voxelization
    sel0 = ME.utils.sparse_quantize(xyz0 / self.voxel_size, return_index=True)
    sel1 = ME.utils.sparse_quantize(xyz1 / self.voxel_size, return_index=True)
    sel_orig = ME.utils.sparse_quantize(xyz_orig / self.voxel_size, return_index=True)

    # Make point clouds using voxelized points
    pcd0 = make_open3d_point_cloud(xyz0)
    pcd1 = make_open3d_point_cloud(xyz1)
    pcd_orig = make_open3d_point_cloud(xyz_orig)

    # Select features and points using the returned voxelized indices
    pcd0.colors = o3d.utility.Vector3dVector(color0[sel0])
    pcd1.colors = o3d.utility.Vector3dVector(color1[sel1])
    pcd_orig.colors = o3d.utility.Vector3dVector(color_orig[sel_orig])

    pcd0.points = o3d.utility.Vector3dVector(np.array(pcd0.points)[sel0])
    pcd1.points = o3d.utility.Vector3dVector(np.array(pcd1.points)[sel1])
    pcd_orig.points = o3d.utility.Vector3dVector(np.array(pcd_orig.points)[sel_orig])

    # print("how many points")
    # print(len(xyz0))
    # print(len(pcd0.points))

    # Get matches
    # NB(s9xie): we probably can reuse the function here; just be careful with the matching size
    # TODO(s9xie): test when voxelsize is small, if we can get diagonal matching matrix.
    # ^ Verified
    matches = get_matching_indices(pcd0, pcd1, trans, matching_search_voxel_size)
    # print(matches[0:10])
    
    # Get features
    npts0 = len(pcd0.colors)
    npts1 = len(pcd1.colors)
    npts_orig = len(pcd_orig.colors)


    feats_train0, feats_train1, feats_train_orig = [], [], []

    feats_train0.append(np.ones((npts0, 3)))
    feats_train1.append(np.ones((npts1, 3)))
    feats_train_orig.append(np.ones((npts_orig, 3)))

    feats0 = np.hstack(feats_train0)
    feats1 = np.hstack(feats_train1)
    feats_orig = np.hstack(feats_train_orig)

    # Get coords
    xyz0 = np.array(pcd0.points)
    xyz1 = np.array(pcd1.points)
    xyz_orig = np.array(pcd_orig.points)

    coords0 = np.floor(xyz0 / self.voxel_size)
    coords1 = np.floor(xyz1 / self.voxel_size)
    coords_orig = np.floor(xyz_orig / self.voxel_size)


    if self.transform:
      coords0, feats0 = self.transform(coords0, feats0)
      coords1, feats1 = self.transform(coords1, feats1)
      coords_orig, feats_orig = self.transform(coords_orig, feats_orig)

    return (xyz0, xyz1, coords0, coords1, feats0, feats1, matches, trans, 
            label, xyz_orig, coords_orig, feats_orig)


class WIPShapeNetDataset(SingleDataset):
  AUGMENT = None

  def __init__(self,
               phase,
               transform=None,
               random_rotation=True,
               random_scale=True,
               manual_seed=False,
               config=None):
    SingleDataset.__init__(self, phase, transform, random_rotation, random_scale,
                         manual_seed, config)
    self.root = root = config.shapenet_dir
    self.phase = phase
    self.npoints = config.shapenet_npoints
    logging.info(f"ScanNet Loading the subset {phase} from {root}")
    self.normalize = True

    # NB(s9xie): train on trainval set and val on test set
    if phase == "train":
        self.train_files = np.load(os.path.join(self.root, "new_train_files.npy"), allow_pickle=True)
        self.val_files = np.load(os.path.join(self.root, "new_val_files.npy"), allow_pickle=True)
        self.files = np.concatenate([self.train_files, self.val_files])
    elif phase == "val":
        self.files = np.load(os.path.join(self.root, "new_test_files.npy"), allow_pickle=True)
    print(self.files[0])
    # TODO(s9xie): we don't need to load labels for now
    # self.labels = []
    # for data_path in tqdm(self.files, total=len(self.files), desc="dataset: retrieve all labels"):
    #     example = np.load(data_path, allow_pickle=True)
    #     self.labels.append(example["cid"])
    # self.label_dataset_size = len(self.files)
    
  def __getitem__(self, idx):
    data0 = np.load(self.files[idx], allow_pickle=True)
    label = data0["cid"]
    xyz0 = data0["point_cloud"]
    if self.phase == "train":
      choice = np.random.choice(xyz0.shape[0], self.npoints, replace=False)
      xyz0 = xyz0[choice, :]
    else:
      xyz0 = xyz0[:self.npoints, :]
    if self.normalize:
      xyz0 = self.pc_normalize(xyz0)
    

    xyz1 = copy.deepcopy(xyz0)
    xyz_orig = copy.deepcopy(xyz0)

    #color0 = data0["color"]
    # print(xyz0.shape)
    color0 = np.ones([xyz0.shape[0], 3])
    color1 = copy.deepcopy(color0)
    color_orig = copy.deepcopy(color0)
    # print(color0.shape)
    
    #############
    matching_search_voxel_size = self.matching_search_voxel_size

    sel_orig = ME.utils.sparse_quantize(xyz_orig / self.voxel_size, return_index=True)
    xyz_orig = xyz_orig[sel_orig]

    pcd_orig = make_open3d_point_cloud(xyz_orig)
    coords_orig = np.floor(np.array(pcd_orig.points) / self.voxel_size)

    pcd_orig.colors = o3d.utility.Vector3dVector(color_orig[sel_orig])
    pcd_orig.points = o3d.utility.Vector3dVector(np.array(pcd_orig.points))

    matches = get_self_matching_indices(pcd_orig, matching_search_voxel_size)

    T0 = sample_random_trans(coords_orig, self.randg, self.rotation_range)
    T1 = sample_random_trans(coords_orig, self.randg, self.rotation_range)

    trans = T1 @ np.linalg.inv(T0)

    coords0 = self.apply_transform(coords_orig, T0)
    coords1 = self.apply_transform(coords_orig, T1)

    if self.random_scale: #and random.random() < 0.95:
      scale0 = self.min_scale + \
          (self.max_scale - self.min_scale) * random.random()

      scale1 = self.min_scale + \
          (self.max_scale - self.min_scale) * random.random()

      xyz0 = scale0 * xyz0
      xyz1 = scale1 * xyz1

    sel0 = ME.utils.sparse_quantize(xyz0 / self.voxel_size, return_index=True)
    sel1 = ME.utils.sparse_quantize(xyz1 / self.voxel_size, return_index=True)

    pcd0 = make_open3d_point_cloud(xyz0)
    pcd1 = make_open3d_point_cloud(xyz1)

    pcd0.colors = o3d.utility.Vector3dVector(color0[sel0])
    pcd1.colors = o3d.utility.Vector3dVector(color1[sel1])

    pcd0.points = o3d.utility.Vector3dVector(np.array(pcd0.points)[sel0])
    pcd1.points = o3d.utility.Vector3dVector(np.array(pcd1.points)[sel1])

    # Get features
    npts0 = len(pcd0.colors)
    npts1 = len(pcd1.colors)
    npts_orig = len(pcd_orig.colors)

    feats_train0, feats_train1, feats_train_orig = [], [], []

    feats_train0.append(np.ones((npts0, 1)))
    feats_train1.append(np.ones((npts1, 1)))
    feats_train_orig.append(np.ones((npts_orig, 1)))

    feats0 = np.hstack(feats_train0)
    feats1 = np.hstack(feats_train1)
    feats_orig = np.hstack(feats_train_orig)

    # Get coords
    xyz0 = np.array(pcd0.points)
    xyz1 = np.array(pcd1.points)
    xyz_orig = np.array(pcd_orig.points)

    coords0 = np.floor(xyz0 / self.voxel_size)
    coords1 = np.floor(xyz1 / self.voxel_size)
    coords_orig = np.floor(xyz_orig / self.voxel_size)


    if self.transform:
      coords0, feats0 = self.transform(coords0, feats0)
      coords1, feats1 = self.transform(coords1, feats1)
      coords_orig, feats_orig = self.transform(coords_orig, feats_orig)

    return (xyz0, xyz1, coords0, coords1, feats0, feats1, matches, trans, 
            label, xyz_orig, coords_orig, feats_orig)


ALL_DATASETS = [ThreeDMatchPairDataset, ShapeNetDataset]
dataset_str_mapping = {d.__name__: d for d in ALL_DATASETS}


def make_data_loader(config, phase, batch_size, num_threads=0, shuffle=None):
  assert phase in ['train', 'trainval', 'val', 'test']
  if shuffle is None:
    shuffle = phase != 'test'

  if config.dataset not in dataset_str_mapping.keys():
    logging.error(f'Dataset {config.dataset}, does not exists in ' +
                  ', '.join(dataset_str_mapping.keys()))

  Dataset = dataset_str_mapping[config.dataset]
  

  use_random_scale = True
  use_random_rotation = True
  if config.dataset == "ShapeNetDataset":
    use_random_scale = False
    use_random_rotation = False
  
  
  transforms = []
  if phase in ['train', 'trainval']:
    use_random_rotation = config.use_random_rotation
    use_random_scale = config.use_random_scale
    transforms += [t.Jitter()]

  dset = Dataset(
      phase,
      transform=t.Compose(transforms),
      random_scale=use_random_scale,
      random_rotation=use_random_rotation,
      config=config)

  if config.dataset != "ShapeNetDataset":
      print("Not ShapeNetDataset")
      collate_pair_fn = default_collate_pair_fn
  else:
      collate_pair_fn = shapenet_collate_pair_fn

  loader = torch.utils.data.DataLoader(
      dset,
      batch_size=batch_size,
      shuffle=shuffle,
      num_workers=num_threads,
      collate_fn=collate_pair_fn,
      pin_memory=False,
      drop_last=True)

  return loader
