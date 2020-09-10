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

from util.pointcloud import get_matching_indices, make_open3d_point_cloud
import lib.transforms as t

import MinkowskiEngine as ME

import open3d as o3d

from torch.utils.data.sampler import RandomSampler
from lib.dataloader import DistributedInfSampler

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

    # Move batchids to the beginning
    xyz_batch0.append(torch.from_numpy(xyz0[batch_id]))
    coords_batch0.append(
        torch.cat((torch.ones(N0, 1).int() * batch_id, 
                   torch.from_numpy(coords0[batch_id]).int()), 1))
    feats_batch0.append(torch.from_numpy(feats0[batch_id]))

    xyz_batch1.append(torch.from_numpy(xyz1[batch_id]))
    coords_batch1.append(
        torch.cat((torch.ones(N1, 1).int() * batch_id, 
                   torch.from_numpy(coords1[batch_id]).int()), 1))
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

# Rotation matrix along axis with angle theta
def M(axis, theta):
  return expm(np.cross(np.eye(3), axis / norm(axis) * theta))

def sample_random_rts(pcd, randg, rotation_range=360, scale_range=None, translation_range=None):
  # first minus mean (centering), then rotation 360 degrees
  T_rot = np.eye(4)
  R = M(randg.rand(3) - 0.5, rotation_range * np.pi / 180.0 * (randg.rand(1) - 0.5))
  T_rot[:3, :3] = R
  T_rot[:3, 3] = R.dot(-np.mean(pcd, axis=0))
  
  T_scale = np.eye(4)
  if scale_range:
      scale = np.random.uniform(*scale_range)
      np.fill_diagonal(T_scale[:3, :3], scale)
  
  T_translation = np.eye(4)
  if translation_range:
    offside = np.random.uniform(*translation_range, 3)
    T_translation[:3, 3] = offside

  return T_rot @ T_scale @ T_translation

def sample_random_trans(pcd, randg, rotation_range=360):
  T = np.eye(4)
  R = M(randg.rand(3) - 0.5, rotation_range * np.pi / 180.0 * (randg.rand(1) - 0.5))
  T[:3, :3] = R
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
    logging.info(f"Loading the subset {phase} from {root}")

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
    # print(len(matches))
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
    # print("after_voxliaze: ", coords0.shape)

    if self.transform:
      coords0, feats0 = self.transform(coords0, feats0)
      coords1, feats1 = self.transform(coords1, feats1)

    return (xyz0, xyz1, coords0, coords1, feats0, feats1, matches, trans)


class ScanNetIndoorPairDataset(PairDataset):
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
    if phase == "train":
        self.root_filelist = root = config.scannet_match_dir
        self.root = '/'
    elif phase == "val":
        self.root = root = config.threed_match_dir
    logging.info(f"Loading the subset {phase} from {root}")
    print("[PHASE]: ##########", phase) 
    if phase == "train":
       fname_txt = self.root_filelist
       with open(fname_txt) as f:
         content = f.readlines()
       fnames = [x.strip().split() for x in content]
       for fname in fnames:
         self.files.append([fname[0], fname[1]])
       
       print("pretraining dataset size:", len(self.files))
       if config.subset_length > 0:
           self.files = random.sample(self.files, config.subset_length)
           print("Using pretraining dataset subset size:", len(self.files))
       else:
           print("Using all pretraining dataset size:", len(self.files))

    elif phase == "val":
        # reuse 3dmatch data
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
    else:
        raise NotImplementedError

  def __getitem__(self, idx):
    file0 = os.path.join(self.root, self.files[idx][0])
    file1 = os.path.join(self.root, self.files[idx][1])
    data0 = np.load(file0)
    data1 = np.load(file1)
    xyz0 = data0["pcd"]
    xyz1 = data1["pcd"]
    color0 = np.ones((xyz0.shape[0], 3)) # data0["color"]
    color1 = np.ones((xyz1.shape[0], 3)) # data0["color"]
    # color1 = data1["color"]
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
    # print(len(matches))
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
    # print("after_voxliaze: ", coords0.shape)

    if self.transform:
      coords0, feats0 = self.transform(coords0, feats0)
      coords1, feats1 = self.transform(coords1, feats1)

    # NB(s9xie): xyz are coordinates in the original system;
    # coords are sparse conv grid coords. (subject to a scaling factor)
    # coords0 -> sinput0_C
    # trans is T0*T1^-1
    return (xyz0, xyz1, coords0, coords1, feats0, feats1, matches, trans)


class ScanNetHardIndoorPairDataset(PairDataset):
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
    if phase == "train":
        self.root_filelist = root = config.scannet_match_dir
        self.root = '/'
    elif phase == "val":
        self.root = root = config.threed_match_dir
    logging.info(f"Loading the subset {phase} from {root}")
    print("[PHASE]: ##########", phase) 
    
    self.scale_range = (0.8, 1.2)#self.SCALE_AUGMENTATION_BOUND
    print("scale_range", self.scale_range)
    
    self.translation_range = (-0.2, 0.2) #self.TRANSLATION_AUGMENTATION_RATIO_BOUND
    print("translation_range", self.translation_range)
    
    if phase == "train":
       fname_txt = self.root_filelist
       with open(fname_txt) as f:
         content = f.readlines()
       fnames = [x.strip().split() for x in content]
       for fname in fnames:
         self.files.append([fname[0], fname[1]])
       print("pretraining dataset size:", len(self.files))

    elif phase == "val":
        # reuse 3dmatch data
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
    else:
        raise NotImplementedError

  def __getitem__(self, idx):
    file0 = os.path.join(self.root, self.files[idx][0])
    file1 = os.path.join(self.root, self.files[idx][1])
    data0 = np.load(file0)
    data1 = np.load(file1)
    xyz0 = data0["pcd"]
    xyz1 = data1["pcd"]
    color0 = np.ones((xyz0.shape[0], 3)) # data0["color"]
    color1 = np.ones((xyz1.shape[0], 3)) # data0["color"]
    # color1 = data1["color"]
    matching_search_voxel_size = self.matching_search_voxel_size

    # if self.random_scale and random.random() < 0.95:
    #   scale = self.min_scale + \
    #       (self.max_scale - self.min_scale) * random.random()
    #   matching_search_voxel_size *= scale
    #   xyz0 = scale * xyz0
    #   xyz1 = scale * xyz1

    if self.random_rotation:
      # T0 = sample_random_trans(xyz0, self.randg, self.rotation_range)
      # T1 = sample_random_trans(xyz1, self.randg, self.rotation_range)
      T0 = sample_random_rts(xyz0, self.randg, self.rotation_range, self.scale_range, self.translation_range)
      T1 = sample_random_rts(xyz1, self.randg, self.rotation_range, self.scale_range, self.translation_range)
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
    # print(len(matches))
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
    # print("after_voxliaze: ", coords0.shape)

    if self.transform:
      coords0, feats0 = self.transform(coords0, feats0)
      coords1, feats1 = self.transform(coords1, feats1)

    # NB(s9xie): xyz are coordinates in the original system;
    # coords are sparse conv grid coords. (subject to a scaling factor)
    # coords0 -> sinput0_C
    # trans is T0*T1^-1
    return (xyz0, xyz1, coords0, coords1, feats0, feats1, matches, trans)


class ScanNetMatchPairDataset(ScanNetIndoorPairDataset):
  OVERLAP_RATIO = 0.3
  DATA_FILES = {
      'train': './config/train_scannet.txt',
      'val': './config/val_3dmatch.txt',
  }

class ScanNetHardMatchPairDataset(ScanNetHardIndoorPairDataset):
  OVERLAP_RATIO = 0.3
  DATA_FILES = {
      'train': './config/train_scannet.txt',
      'val': './config/val_3dmatch.txt',
  }

ALL_DATASETS = [ScanNetMatchPairDataset, ScanNetHardMatchPairDataset]
dataset_str_mapping = {d.__name__: d for d in ALL_DATASETS}


def make_data_loader(config, phase, batch_size, num_threads=0, shuffle=None, inf_sample=False):
  assert phase in ['train', 'trainval', 'val', 'test']
  if shuffle is None:
    shuffle = phase != 'test'

  if config.dataset not in dataset_str_mapping.keys():
    logging.error(f'Dataset {config.dataset}, does not exists in ' +
                  ', '.join(dataset_str_mapping.keys()))

  Dataset = dataset_str_mapping[config.dataset]


  use_random_scale = True
  use_random_rotation = True

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
  print(dset)
  collate_pair_fn = default_collate_pair_fn
  
  print("original batch_size=", batch_size)
  batch_size = batch_size // config.num_gpus

  if config.num_gpus > 1:
    sampler = DistributedInfSampler(dset)
  else:
    sampler = None
  
  loader = torch.utils.data.DataLoader(
      dset,
      batch_size=batch_size,
      shuffle=False if sampler else shuffle,
      num_workers=num_threads,
      collate_fn=collate_pair_fn,
      pin_memory=False,
      sampler=sampler,
      drop_last=True)

  return loader
