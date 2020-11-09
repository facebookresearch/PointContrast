# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from lib.datasets import stanford 
from lib.datasets import scannet

DATASETS = []


def add_datasets(module):
  DATASETS.extend([getattr(module, a) for a in dir(module) if 'Dataset' in a])


add_datasets(stanford)
add_datasets(scannet)


def load_dataset(name):
  '''Creates and returns an instance of the datasets given its name.
  '''
  # Find the model class from its name
  mdict = {dataset.__name__: dataset for dataset in DATASETS}
  if name not in mdict:
    print('Invalid dataset index. Options are:')
    # Display a list of valid dataset names
    for dataset in DATASETS:
      print('\t* {}'.format(dataset.__name__))
    raise ValueError(f'Dataset {name} not defined')
  DatasetClass = mdict[name]

  return DatasetClass
