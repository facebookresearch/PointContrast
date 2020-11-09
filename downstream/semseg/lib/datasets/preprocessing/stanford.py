# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import glob
import numpy as np
import os

from tqdm import tqdm

from lib.utils import mkdir_p
from lib.pc_utils import save_point_cloud

import MinkowskiEngine as ME

STANFORD_3D_IN_PATH = 'YOUR_DATASET_PATH_HERE'
STANFORD_3D_OUT_PATH = 'YOUR_PREPROCESSED_DATASET_PATH_HERE'

STANFORD_3D_TO_SEGCLOUD_LABEL = {
    4: 0,
    8: 1,
    12: 2,
    1: 3,
    6: 4,
    13: 5,
    7: 6,
    5: 7,
    11: 8,
    3: 9,
    9: 10,
    2: 11,
    0: 12,
}


class Stanford3DDatasetConverter:

  CLASSES = [
      'clutter', 'beam', 'board', 'bookcase', 'ceiling', 'chair', 'column', 'door', 'floor', 'sofa',
      'stairs', 'table', 'wall', 'window'
  ]
  TRAIN_TEXT = 'train'
  VAL_TEXT = 'val'
  TEST_TEXT = 'test'

  @classmethod
  def read_txt(cls, txtfile):
    # Read txt file and parse its content.
    with open(txtfile) as f:
      pointcloud = []
      for l in f:
        try:
          pointcloud += [[float(li) for li in l.split()]]
        except Exception as e:
          print(e, txtfile)
          continue
      # pointcloud = [l.split() for l in f]

    # Load point cloud to named numpy array.
    pointcloud = np.array(pointcloud).astype(np.float32)
    assert pointcloud.shape[1] == 6
    xyz = pointcloud[:, :3].astype(np.float32)
    rgb = pointcloud[:, 3:].astype(np.uint8)
    return xyz, rgb

  @classmethod
  def convert_to_ply(cls, root_path, out_path):
    """Convert Stanford3DDataset to PLY format that is compatible with
    Synthia dataset. Assumes file structure as given by the dataset.
    Outputs the processed PLY files to `STANFORD_3D_OUT_PATH`.
    """

    txtfiles = glob.glob(os.path.join(root_path, '*/*/*.txt'))
    for txtfile in tqdm(txtfiles):
      file_sp = os.path.normpath(txtfile).split(os.path.sep)
      target_path = os.path.join(out_path, file_sp[-3])
      out_file = os.path.join(target_path, file_sp[-2] + '.ply')

      if os.path.exists(out_file):
        print(out_file, ' exists')
        continue

      annotation, _ = os.path.split(txtfile)
      subclouds = glob.glob(os.path.join(annotation, 'Annotations/*.txt'))
      coords, feats, labels = [], [], []
      for inst, subcloud in enumerate(subclouds):
        # Read ply file and parse its rgb values.
        xyz, rgb = cls.read_txt(subcloud)
        _, annotation_subfile = os.path.split(subcloud)
        clsidx = cls.CLASSES.index(annotation_subfile.split('_')[0])

        coords.append(xyz)
        feats.append(rgb)
        labels.append(np.ones((len(xyz), 1), dtype=np.int32) * clsidx)

      if len(coords) == 0:
        print(txtfile, ' has 0 files.')
      else:
        # Concat
        coords = np.concatenate(coords, 0)
        feats = np.concatenate(feats, 0)
        labels = np.concatenate(labels, 0)
        inds, collabels = ME.utils.sparse_quantize(
            coords,
            feats,
            labels,
            return_index=True,
            ignore_label=255,
            quantization_size=0.01  # 1cm
        )
        pointcloud = np.concatenate((coords[inds], feats[inds], collabels[:, None]), axis=1)

        # Write ply file.
        mkdir_p(target_path)
        save_point_cloud(pointcloud, out_file, with_label=True, verbose=False)


def generate_splits(stanford_out_path):
  """Takes preprocessed out path and generate txt files"""
  split_path = './splits/stanford'
  mkdir_p(split_path)
  for i in range(1, 7):
    curr_path = os.path.join(stanford_out_path, f'Area_{i}')
    files = glob.glob(os.path.join(curr_path, '*.ply'))
    files = [os.path.relpath(full_path, stanford_out_path) for full_path in files]
    out_txt = os.path.join(split_path, f'area{i}.txt')
    with open(out_txt, 'w') as f:
      f.write('\n'.join(files))


if __name__ == '__main__':
  Stanford3DDatasetConverter.convert_to_ply(STANFORD_3D_IN_PATH, STANFORD_3D_OUT_PATH)
  generate_splits(STANFORD_3D_OUT_PATH)
