from abc import ABC
from pathlib import Path
from collections import defaultdict
import copy

import random
import numpy as np
import os
from enum import Enum

from torch.utils.data import Dataset, DataLoader

import MinkowskiEngine as ME

from lib.pc_utils import read_plyfile
from plyfile import PlyData

import lib.transforms as t
import lib.transforms_scannet as tsc
import lib.transforms_pointnet as psc
from torchvision import transforms as torchvision_transforms

from lib.dataloader import InfSampler, DistributedInfSampler
from lib.voxelizer import Voxelizer

import open3d as o3d
from scipy.linalg import expm, norm
from util.pointcloud import get_matching_indices, get_matching_indices_on_voxels, make_open3d_point_cloud

import logging
import torch

class DatasetPhase(Enum):
  Train = 0
  Val = 1
  Val2 = 2
  TrainVal = 3
  Test = 4


def datasetphase_2str(arg):
  if arg == DatasetPhase.Train:
    return 'train'
  elif arg == DatasetPhase.Val:
    return 'val'
  elif arg == DatasetPhase.Val2:
    return 'val2'
  elif arg == DatasetPhase.TrainVal:
    return 'trainval'
  elif arg == DatasetPhase.Test:
    return 'test'
  else:
    raise ValueError('phase must be one of dataset enum.')


def str2datasetphase_type(arg):
  if arg.upper() == 'TRAIN':
    return DatasetPhase.Train
  elif arg.upper() == 'VAL':
    return DatasetPhase.Val
  elif arg.upper() == 'VAL2':
    return DatasetPhase.Val2
  elif arg.upper() == 'TRAINVAL':
    return DatasetPhase.TrainVal
  elif arg.upper() == 'TEST':
    return DatasetPhase.Test
  else:
    raise ValueError('phase must be one of train/val/test')


def cache(func):

  def wrapper(self, *args, **kwargs):
    # Assume that args[0] is index
    index = args[0]
    if self.cache:
      if index not in self.cache_dict[func.__name__]:
        results = func(self, *args, **kwargs)
        self.cache_dict[func.__name__][index] = results
      return self.cache_dict[func.__name__][index]
    else:
      return func(self, *args, **kwargs)

  return wrapper


class DictDataset(Dataset, ABC):

  IS_FULL_POINTCLOUD_EVAL = False

  def __init__(self,
               data_paths,
               prevoxel_transform=None,
               input_transform=None,
               target_transform=None,
               cache=False,
               data_root='/'):
    """
    data_paths: list of lists, [[str_path_to_input, str_path_to_label], [...]]
    """
    Dataset.__init__(self)

    # Allows easier path concatenation
    if not isinstance(data_root, Path):
      data_root = Path(data_root)

    self.data_root = data_root
    self.data_paths = sorted(data_paths)

    self.prevoxel_transform = prevoxel_transform
    self.input_transform = input_transform
    self.target_transform = target_transform

    # dictionary of input
    self.data_loader_dict = {
        'input': (self.load_input, self.input_transform),
        'target': (self.load_target, self.target_transform)
    }

    # For large dataset, do not cache
    self.cache = cache
    self.cache_dict = defaultdict(dict)
    self.loading_key_order = ['input', 'target']

  def load_input(self, index):
    raise NotImplementedError

  def load_target(self, index):
    raise NotImplementedError

  def get_classnames(self):
    pass

  def reorder_result(self, result):
    return result

  def __getitem__(self, index):
    out_array = []
    for k in self.loading_key_order:
      loader, transformer = self.data_loader_dict[k]
      v = loader(index)
      if transformer:
        v = transformer(v)
      out_array.append(v)
    return out_array

  def __len__(self):
    return len(self.data_paths)


class VoxelizationDatasetBase(DictDataset, ABC):
  IS_TEMPORAL = False
  CLIP_BOUND = (-1000, -1000, -1000, 1000, 1000, 1000)
  ROTATION_AXIS = None
  NUM_IN_CHANNEL = None
  # HACK(s9xie): do not use RGB, use points only
  # NUM_IN_CHANNEL = 1
  NUM_LABELS = -1  # Number of labels in the dataset, including all ignore classes
  IGNORE_LABELS = None  # labels that are not evaluated

  def __init__(self,
               data_paths,
               prevoxel_transform=None,
               input_transform=None,
               target_transform=None,
               cache=False,
               data_root='/',
               ignore_mask=255,
               return_transformation=False,
               **kwargs):
    """
    ignore_mask: label value for ignore class. It will not be used as a class in the loss or evaluation.
    """
    DictDataset.__init__(
        self,
        data_paths,
        prevoxel_transform=prevoxel_transform,
        input_transform=input_transform,
        target_transform=target_transform,
        cache=cache,
        data_root=data_root)

    self.ignore_mask = ignore_mask
    self.return_transformation = return_transformation

  def __getitem__(self, index):
    raise NotImplementedError

  def load_ply_old(self, index):
    filepath = self.data_root / self.data_paths[index]
    return read_plyfile(filepath), None

  def load_ply(self, index):
    filepath = self.data_root / self.data_paths[index]
    plydata = PlyData.read(filepath)
    data = plydata.elements[0].data
    coords = np.array([data['x'], data['y'], data['z']], dtype=np.float32).T
    feats = np.array([data['red'], data['green'], data['blue']], dtype=np.float32).T
    labels = np.array(data['label'], dtype=np.int32)
    return coords, feats, labels, None

  def __len__(self):
    num_data = len(self.data_paths)
    return num_data


class VoxelizationDataset(VoxelizationDatasetBase):
  """This dataset loads RGB point clouds and their labels as a list of points
  and voxelizes the pointcloud with sufficient data augmentation.
  """
  # Voxelization arguments
  VOXEL_SIZE = 0.05  # 5cm

  # Coordinate Augmentation Arguments: Unlike feature augmentation, coordinate
  # augmentation has to be done before voxelization
  SCALE_AUGMENTATION_BOUND = (0.9, 1.1)
  ROTATION_AUGMENTATION_BOUND = ((-np.pi / 6, np.pi / 6), (-np.pi, np.pi), (-np.pi / 6, np.pi / 6))
  TRANSLATION_AUGMENTATION_RATIO_BOUND = ((-0.2, 0.2), (-0.05, 0.05), (-0.2, 0.2))
  ELASTIC_DISTORT_PARAMS = None

  # MISC.
  PREVOXELIZE_VOXEL_SIZE = None

  def __init__(self,
               data_paths,
               prevoxel_transform=None,
               input_transform=None,
               target_transform=None,
               data_root='/',
               ignore_label=255,
               return_transformation=False,
               augment_data=False,
               config=None,
               **kwargs):

    self.augment_data = augment_data
    self.config = config
    VoxelizationDatasetBase.__init__(
        self,
        data_paths,
        prevoxel_transform=prevoxel_transform,
        input_transform=input_transform,
        target_transform=target_transform,
        cache=cache,
        data_root=data_root,
        ignore_mask=ignore_label,
        return_transformation=return_transformation)

    # Prevoxel transformations
    self.voxelizer = Voxelizer(
        voxel_size=self.VOXEL_SIZE,
        clip_bound=self.CLIP_BOUND,
        use_augmentation=augment_data,
        scale_augmentation_bound=self.SCALE_AUGMENTATION_BOUND,
        rotation_augmentation_bound=self.ROTATION_AUGMENTATION_BOUND,
        translation_augmentation_ratio_bound=self.TRANSLATION_AUGMENTATION_RATIO_BOUND,
        ignore_label=ignore_label)

    # map labels not evaluated to ignore_label
    label_map = {}
    n_used = 0
    for l in range(self.NUM_LABELS):
      if l in self.IGNORE_LABELS:
        label_map[l] = self.ignore_mask
      else:
        label_map[l] = n_used
        n_used += 1
    label_map[self.ignore_mask] = self.ignore_mask
    self.label_map = label_map
    self.NUM_LABELS -= len(self.IGNORE_LABELS)

  def convert_mat2cfl(self, mat):
    # Generally, xyz,rgb,label
    return mat[:, :3], mat[:, 3:-1], mat[:, -1]

  def __getitem__(self, index):
    pointcloud, center = self.load_ply(index)

    # Downsample the pointcloud with finer voxel size before transformation for memory and speed
    if self.PREVOXELIZE_VOXEL_SIZE is not None:
      inds = ME.utils.sparse_quantize(
          pointcloud[:, :3] / self.PREVOXELIZE_VOXEL_SIZE, return_index=True)
      pointcloud = pointcloud[inds]

    # Prevoxel transformations
    if self.prevoxel_transform is not None:
        pointcloud = self.prevoxel_transform(pointcloud)
    
    coords, feats, labels = self.convert_mat2cfl(pointcloud)
    coords, feats, labels, transformation = self.voxelizer.voxelize(
        coords, feats, labels, center=center)

    # map labels not used for evaluation to ignore_label
    if self.input_transform is not None:
      coords, feats, labels = self.input_transform(coords, feats, labels)
    if self.target_transform is not None:
      coords, feats, labels = self.target_transform(coords, feats, labels)
    if self.IGNORE_LABELS is not None:
      labels = np.array([self.label_map[x] for x in labels], dtype=np.int)

    return_args = [coords, feats, labels]
    if self.return_transformation:
      return_args.extend([pointcloud.astype(np.float32), transformation.astype(np.float32)])
    return tuple(return_args)


# Rotation matrix along axis with angle theta
def M(axis, theta):
  return expm(np.cross(np.eye(3), axis / norm(axis) * theta))


def sample_random_trans(pcd, randg, rotation_range=360, scale_range=None, translation_range=None):
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

class MatchVoxelizationPairDataset(VoxelizationDatasetBase):
  """This dataset loads RGB point clouds and their labels as a list of points
  and voxelizes the pointcloud with sufficient data augmentation.
  """
  # Voxelization arguments
  VOXEL_SIZE = 0.05  # 5cm

  # Coordinate Augmentation Arguments: Unlike feature augmentation, coordinate
  # augmentation has to be done before voxelization
  SCALE_AUGMENTATION_BOUND = (0.8, 1.2)
  TRANSLATION_AUGMENTATION_RATIO_BOUND = (-0.2, 0.2)
  ELASTIC_DISTORT_PARAMS = None

  # MISC.
  PREVOXELIZE_VOXEL_SIZE = None
  
  def apply_transform(self, pts, trans):
    R = trans[:3, :3]
    T = trans[:3, 3]
    pts = pts @ R.T + T
    return pts

  def __init__(self,
               data_paths,
               prevoxel_transform=None,
               input_transform=None,
               target_transform=None,
               data_root='/',
               ignore_label=255,
               return_transformation=False,
               augment_data=False,
               config=None,
               manual_seed=False,
               phase=None,
               **kwargs):

    self.phase = phase
    self.augment_data = augment_data
    self.config = config
    self.voxel_size = config.voxel_size
    self.free_rot = config.free_rot
    self.matching_search_voxel_size = \
      config.voxel_size * config.positive_pair_search_voxel_size_multiplier
    
    VoxelizationDatasetBase.__init__(
        self,
        data_paths,
        prevoxel_transform=prevoxel_transform,
        input_transform=input_transform,
        target_transform=target_transform,
        cache=cache,
        data_root=data_root,
        ignore_mask=ignore_label,
        return_transformation=return_transformation)

    # map labels not evaluated to ignore_label
    label_map = {}
    n_used = 0
    for l in range(self.NUM_LABELS):
      if l in self.IGNORE_LABELS:
        label_map[l] = self.ignore_mask
      else:
        label_map[l] = n_used
        n_used += 1
    label_map[self.ignore_mask] = self.ignore_mask
    self.label_map = label_map
    self.NUM_LABELS -= len(self.IGNORE_LABELS)

    # TODO(s9xie): move it
    self.use_color_feat = config.use_color_feat
    self.use_random_crop = config.use_random_crop
    self.min_scale = config.min_scale
    self.max_scale = config.max_scale
    self.rotation_range = config.rotation_range
    # TODO: fix this
    self.scale_range = (0.8, 1.2)#self.SCALE_AUGMENTATION_BOUND
    print("scale_range", self.scale_range)
    self.translation_range = (-0.2, 0.2) #self.TRANSLATION_AUGMENTATION_RATIO_BOUND
    print("translation_range", self.translation_range)

    self.randg = np.random.RandomState()
    if manual_seed:
      self.reset_seed()

  def reset_seed(self, seed=0):
    logging.info(f"Resetting the data loader seed to {seed}")
    self.randg.seed(seed)

  def convert_mat2cfl(self, mat):
    # Generally, xyz,rgb,label
    return mat[:, :3], mat[:, 3:-1], mat[:, -1]
  
  def random_crop(self, xyz, feats, labels):
    bound_min = np.min(xyz, 0).astype(float)
    bound_max = np.max(xyz, 0).astype(float)
    bound_size = bound_max - bound_min
    lim = bound_size / self.config.crop_factor # (2/3)**3 of original size
    while True:
      vertex = random.uniform(bound_min, bound_size - lim)
      clip_inds = ((xyz[:, 0] >= vertex[0]) & \
                   (xyz[:, 0] <= (lim[0] + vertex[0])) & \
                   (xyz[:, 1] >= (vertex[1])) & \
                   (xyz[:, 1] <= (lim[1] + vertex[1])) & \
                   (xyz[:, 2] >= (vertex[2])) & \
                   (xyz[:, 2] <= (lim[2] + vertex[2])))
      if len(xyz[clip_inds]) >= self.config.crop_min_num_points:
        break
    return xyz[clip_inds], feats[clip_inds], labels[clip_inds]

  def __getitem__(self, index):
    xyz0, feats0, labels0, center = self.load_ply(index)
    
    train_transforms = psc.Compose(
            [
                # psc.PointcloudToTensor(),
                # psc.PointcloudScale(),
                # psc.PointcloudTranslate(),
                # psc.PointcloudRotate(),
                psc.PointcloudJitter(),
                psc.PointcloudRotatePerturbation(),
                psc.PointcloudRandomInputDropout(),
            ]) 
   
    if self.phase == DatasetPhase.Train and self.use_random_crop:
        # print("before cropping: ", len(xyz0))
        xyz0, feats0, labels0 = self.random_crop(xyz0, feats0, labels0)
        xyz0, feats0, labels0 = train_transforms(xyz0, feats0, labels0)
        # if len(xyz0) < 5000:
        #     print("after cropping: ", len(xyz0))
    xyz1 = copy.deepcopy(xyz0)
    feats1 = copy.deepcopy(feats0)

    T0 = sample_random_trans(xyz0, self.randg, self.rotation_range, self.scale_range, self.translation_range)
    T1 = sample_random_trans(xyz1, self.randg, self.rotation_range, self.scale_range, self.translation_range)

    trans = T1 @ np.linalg.inv(T0)

    xyz0 = self.apply_transform(xyz0, T0)
    xyz1 = self.apply_transform(xyz1, T1)
    
    matching_search_voxel_size = self.matching_search_voxel_size
    
    # TODO: Make the scaling async
    # TODO: Add translation
    # if random.random() < 0.95:
    #   scale = self.min_scale + \
    #       (self.max_scale - self.min_scale) * random.random()
    #   matching_search_voxel_size *= scale
    #   xyz0 = scale * xyz0
    #   xyz1 = scale * xyz1

    # Voxelization
    sel0 = ME.utils.sparse_quantize(xyz0 / self.voxel_size, return_index=True)
    sel1 = ME.utils.sparse_quantize(xyz1 / self.voxel_size, return_index=True)

    # Make point clouds using voxelized points
    pcd0 = make_open3d_point_cloud(xyz0)
    pcd1 = make_open3d_point_cloud(xyz1)

    # Select features and points using the returned voxelized indices
    pcd0.colors = o3d.utility.Vector3dVector(feats0[sel0])
    pcd1.colors = o3d.utility.Vector3dVector(feats1[sel1])

    pcd0.points = o3d.utility.Vector3dVector(np.array(pcd0.points)[sel0])
    pcd1.points = o3d.utility.Vector3dVector(np.array(pcd1.points)[sel1])

    matches = get_matching_indices(pcd0, pcd1, trans, matching_search_voxel_size)
    
    
    if self.use_color_feat:
      # Get features
      feats0 = feats0[sel0]
      feats1 = feats1[sel1]
    else:
      npts0 = len(feats0[sel0])
      npts1 = len(feats1[sel1])
      feats_train0, feats_train1 = [], []
      feats_train0.append(np.ones((npts0, 3)))
      feats_train1.append(np.ones((npts1, 3)))
      feats0 = np.hstack(feats_train0)
      feats1 = np.hstack(feats_train1)
    # feats_train0, feats_train1 = [], []

    # feats_train0.append(np.ones((npts0, 3)))
    # feats_train1.append(np.ones((npts1, 3)))

    # feats0 = np.hstack(feats_train0)
    # feats1 = np.hstack(feats_train1)

    # Get coords
    xyz0 = np.array(pcd0.points)
    xyz1 = np.array(pcd1.points)
    
    # Shall we apply aug here???? Jittering and dropping out, maybe rotation too???

    coords0 = np.floor(xyz0 / self.voxel_size)
    coords1 = np.floor(xyz1 / self.voxel_size)

    self_transform = t.Compose([t.Jitter()])
    coords0, feats0 = self_transform(coords0, feats0)
    coords1, feats1 = self_transform(coords1, feats1)

    return (xyz0, xyz1, coords0, coords1, feats0, feats1, matches, trans, 
            labels0)


class VoxelizationPairDataset(VoxelizationDatasetBase):
  """This dataset loads RGB point clouds and their labels as a list of points
  and voxelizes the pointcloud with sufficient data augmentation.
  """
  # Voxelization arguments
  VOXEL_SIZE = 0.05  # 5cm

  # Coordinate Augmentation Arguments: Unlike feature augmentation, coordinate
  # augmentation has to be done before voxelization
  SCALE_AUGMENTATION_BOUND = (0.9, 1.1)
  ROTATION_AUGMENTATION_BOUND = ((-np.pi / 6, np.pi / 6), (-np.pi, np.pi), (-np.pi / 6, np.pi / 6))
  TRANSLATION_AUGMENTATION_RATIO_BOUND = ((-0.2, 0.2), (-0.05, 0.05), (-0.2, 0.2))
  ELASTIC_DISTORT_PARAMS = None

  # MISC.
  PREVOXELIZE_VOXEL_SIZE = None

  def __init__(self,
               data_paths,
               prevoxel_transform=None,
               input_transform=None,
               target_transform=None,
               data_root='/',
               ignore_label=255,
               return_transformation=False,
               augment_data=False,
               config=None,
               manual_seed=False,
               **kwargs):

    self.augment_data = augment_data
    self.config = config
    self.voxel_size = config.voxel_size
    self.free_rot = config.free_rot
    self.matching_search_voxel_size = \
      config.voxel_size * config.positive_pair_search_voxel_size_multiplier

    VoxelizationDatasetBase.__init__(
        self,
        data_paths,
        prevoxel_transform=prevoxel_transform,
        input_transform=input_transform,
        target_transform=target_transform,
        cache=cache,
        data_root=data_root,
        ignore_mask=ignore_label,
        return_transformation=return_transformation)

    # Prevoxel transformations
    self.voxelizer = Voxelizer(
        voxel_size=self.VOXEL_SIZE,
        clip_bound=self.CLIP_BOUND,
        use_augmentation=augment_data,
        scale_augmentation_bound=self.SCALE_AUGMENTATION_BOUND,
        rotation_augmentation_bound=self.ROTATION_AUGMENTATION_BOUND,
        translation_augmentation_ratio_bound=self.TRANSLATION_AUGMENTATION_RATIO_BOUND,
        ignore_label=ignore_label)

    # map labels not evaluated to ignore_label
    label_map = {}
    n_used = 0
    for l in range(self.NUM_LABELS):
      if l in self.IGNORE_LABELS:
        label_map[l] = self.ignore_mask
      else:
        label_map[l] = n_used
        n_used += 1
    label_map[self.ignore_mask] = self.ignore_mask
    self.label_map = label_map
    self.NUM_LABELS -= len(self.IGNORE_LABELS)

    # TODO(s9xie): move it
    self.use_color_feat = config.use_color_feat

    self.randg = np.random.RandomState()
    if manual_seed:
      self.reset_seed()

  def reset_seed(self, seed=0):
    logging.info(f"Resetting the data loader seed to {seed}")
    self.randg.seed(seed)

  def convert_mat2cfl(self, mat):
    return mat[:, :3], mat[:, 3:-1], mat[:, -1]

  def __getitem__(self, index):
    xyz_orig, feats_orig, labels_orig, center = self.load_ply(index)
    pcd_orig = make_open3d_point_cloud(xyz_orig)
    pcd_orig.colors = o3d.utility.Vector3dVector(feats_orig)
    pcd_orig.points = o3d.utility.Vector3dVector(np.array(pcd_orig.points))

    coords_0, feats_0, labels_0, T0 = self.voxelizer.voxelize(
        xyz_orig, feats_orig, labels_orig, center=center, free_rot=self.free_rot)
    coords_1, feats_1, labels_1, T1 = self.voxelizer.voxelize(
        xyz_orig, feats_orig, labels_orig, center=center, free_rot=self.free_rot)
    
    if self.input_transform is not None:
      coords_0, feats_0, labels_0 = self.input_transform(coords_0, feats_0, labels_0)
      coords_1, feats_1, labels_1 = self.input_transform(coords_1, feats_1, labels_1)

    if self.IGNORE_LABELS is not None:
      labels_0 = np.array([self.label_map[x] for x in labels_0], dtype=np.int)
      labels_1 = np.array([self.label_map[x] for x in labels_1], dtype=np.int)

    trans = T1 @ np.linalg.inv(T0)

    matching_search_voxel_size = self.matching_search_voxel_size
    voxel_size = self.voxel_size
    # Make point clouds using voxelized points
    pcd0 = make_open3d_point_cloud(coords_0)
    pcd1 = make_open3d_point_cloud(coords_1)

    # Select features and points using the returned voxelized indices
    pcd0.colors = o3d.utility.Vector3dVector(feats_0)
    pcd1.colors = o3d.utility.Vector3dVector(feats_1)

    pcd0.points = o3d.utility.Vector3dVector(np.array(pcd0.points))
    pcd1.points = o3d.utility.Vector3dVector(np.array(pcd1.points))
    
    matches = get_matching_indices_on_voxels(pcd0, pcd1, trans, voxel_size, matching_search_voxel_size)
    # Get features
    if self.use_color_feat:
      feats0 = pcd0.colors
      feats1 = pcd0.colors
    else:
      npts0 = len(pcd0.colors)
      npts1 = len(pcd1.colors)
      feats_train0, feats_train1 = [], []
      feats_train0.append(np.ones((npts0, 1)))
      feats_train1.append(np.ones((npts1, 1)))
      feats0 = np.hstack(feats_train0)
      feats1 = np.hstack(feats_train1)

    return (coords_0, coords_1, feats_0, feats_1, matches, trans, labels_0)
