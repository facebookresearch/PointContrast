import logging
import os
import sys
from pathlib import Path

import numpy as np
from scipy import spatial

from lib.dataset import VoxelizationDataset, DatasetPhase, str2datasetphase_type
from lib.pc_utils import read_plyfile, save_point_cloud
from lib.utils import read_txt, fast_hist, per_class_iu

CLASS_LABELS = ('wall', 'floor', 'cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window',
                'bookshelf', 'picture', 'counter', 'desk', 'curtain', 'refrigerator',
                'shower curtain', 'toilet', 'sink', 'bathtub', 'otherfurniture')
VALID_CLASS_IDS = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39)
SCANNET_COLOR_MAP = {
    0: (0., 0., 0.),
    1: (174., 199., 232.),
    2: (152., 223., 138.),
    3: (31., 119., 180.),
    4: (255., 187., 120.),
    5: (188., 189., 34.),
    6: (140., 86., 75.),
    7: (255., 152., 150.),
    8: (214., 39., 40.),
    9: (197., 176., 213.),
    10: (148., 103., 189.),
    11: (196., 156., 148.),
    12: (23., 190., 207.),
    14: (247., 182., 210.),
    15: (66., 188., 102.),
    16: (219., 219., 141.),
    17: (140., 57., 197.),
    18: (202., 185., 52.),
    19: (51., 176., 203.),
    20: (200., 54., 131.),
    21: (92., 193., 61.),
    22: (78., 71., 183.),
    23: (172., 114., 82.),
    24: (255., 127., 14.),
    25: (91., 163., 138.),
    26: (153., 98., 156.),
    27: (140., 153., 101.),
    28: (158., 218., 229.),
    29: (100., 125., 154.),
    30: (178., 127., 135.),
    32: (146., 111., 194.),
    33: (44., 160., 44.),
    34: (112., 128., 144.),
    35: (96., 207., 209.),
    36: (227., 119., 194.),
    37: (213., 92., 176.),
    38: (94., 106., 211.),
    39: (82., 84., 163.),
    40: (100., 85., 144.),
}


class ScannetVoxelizationDataset(VoxelizationDataset):

  # Voxelization arguments
  CLIP_BOUND = None
  TEST_CLIP_BOUND = None
  VOXEL_SIZE = 0.05

  # Augmentation arguments
  ROTATION_AUGMENTATION_BOUND = ((-np.pi / 64, np.pi / 64), (-np.pi / 64, np.pi / 64), (-np.pi,
                                                                                        np.pi))
  TRANSLATION_AUGMENTATION_RATIO_BOUND = ((-0.2, 0.2), (-0.2, 0.2), (0, 0))
  ELASTIC_DISTORT_PARAMS = ((0.2, 0.4), (0.8, 1.6))

  ROTATION_AXIS = 'z'
  LOCFEAT_IDX = 2
  NUM_LABELS = 41  # Will be converted to 20 as defined in IGNORE_LABELS.
  IGNORE_LABELS = tuple(set(range(41)) - set(VALID_CLASS_IDS))
  IS_FULL_POINTCLOUD_EVAL = True

  # If trainval.txt does not exist, copy train.txt and add contents from val.txt
  DATA_PATH_FILE = {
      DatasetPhase.Train: 'scannetv2_train.txt',
      DatasetPhase.Val: 'scannetv2_val.txt',
      DatasetPhase.TrainVal: 'scannetv2_trainval.txt',
      DatasetPhase.Test: 'scannetv2_test.txt'
  }

  def __init__(self,
               config,
               prevoxel_transform=None,
               input_transform=None,
               target_transform=None,
               augment_data=True,
               elastic_distortion=False,
               cache=False,
               phase=DatasetPhase.Train):
    if isinstance(phase, str):
      phase = str2datasetphase_type(phase)
    # Use cropped rooms for train/val
    data_root = config.scannet_path
    if phase not in [DatasetPhase.Train, DatasetPhase.TrainVal]:
      self.CLIP_BOUND = self.TEST_CLIP_BOUND
    data_paths = read_txt(os.path.join('./splits/scannet', self.DATA_PATH_FILE[phase]))
    logging.info('Loading {}: {}'.format(self.__class__.__name__, self.DATA_PATH_FILE[phase]))
    super().__init__(
        data_paths,
        data_root=data_root,
        prevoxel_transform=prevoxel_transform,
        input_transform=input_transform,
        target_transform=target_transform,
        ignore_label=config.ignore_label,
        return_transformation=config.return_transformation,
        augment_data=augment_data,
        elastic_distortion=elastic_distortion,
        config=config)

  def get_output_id(self, iteration):
    return '_'.join(Path(self.data_paths[iteration]).stem.split('_')[:2])

  def _augment_locfeat(self, pointcloud):
    # Assuming that pointcloud is xyzrgb(...), append location feat.
    pointcloud = np.hstack(
        (pointcloud[:, :6], 100 * np.expand_dims(pointcloud[:, self.LOCFEAT_IDX], 1),
         pointcloud[:, 6:]))
    return pointcloud

  def test_pointcloud(self, pred_dir):
    print('Running full pointcloud evaluation.')
    eval_path = os.path.join(pred_dir, 'fulleval')
    os.makedirs(eval_path, exist_ok=True)
    # Join room by their area and room id.
    # Test independently for each room.
    sys.setrecursionlimit(100000)  # Increase recursion limit for k-d tree.
    hist = np.zeros((self.NUM_LABELS, self.NUM_LABELS))
    for i, data_path in enumerate(self.data_paths):
      room_id = self.get_output_id(i)
      pred = np.load(os.path.join(pred_dir, 'pred_%04d_%02d.npy' % (i, 0)))

      # save voxelized pointcloud predictions
      save_point_cloud(
          np.hstack((pred[:, :3], np.array([SCANNET_COLOR_MAP[i] for i in pred[:, -1]]))),
          f'{eval_path}/{room_id}_voxel.ply',
          verbose=False)

      fullply_f = self.data_root / data_path
      query_pointcloud = read_plyfile(fullply_f)
      query_xyz = query_pointcloud[:, :3]
      query_label = query_pointcloud[:, -1]
      # Run test for each room.
      pred_tree = spatial.KDTree(pred[:, :3], leafsize=500)
      _, result = pred_tree.query(query_xyz)
      ptc_pred = pred[result, 3].astype(int)
      # Save prediciton in txt format for submission.
      np.savetxt(f'{eval_path}/{room_id}.txt', ptc_pred, fmt='%i')
      # Save prediciton in colored pointcloud for visualization.
      save_point_cloud(
          np.hstack((query_xyz, np.array([SCANNET_COLOR_MAP[i] for i in ptc_pred]))),
          f'{eval_path}/{room_id}.ply',
          verbose=False)
      # Evaluate IoU.
      if self.IGNORE_LABELS is not None:
        ptc_pred = np.array([self.label_map[x] for x in ptc_pred], dtype=np.int)
        query_label = np.array([self.label_map[x] for x in query_label], dtype=np.int)
      hist += fast_hist(ptc_pred, query_label, self.NUM_LABELS)
    ious = per_class_iu(hist) * 100
    print('mIoU: ' + str(np.nanmean(ious)) + '\n'
          'Class names: ' + ', '.join(CLASS_LABELS) + '\n'
          'IoU: ' + ', '.join(np.round(ious, 2).astype(str)))


class ScannetVoxelization2cmDataset(ScannetVoxelizationDataset):
  VOXEL_SIZE = 0.02
