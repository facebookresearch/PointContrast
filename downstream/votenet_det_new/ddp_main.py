# Copyright (c) Facebook, Inc. and its affiliates.
#  
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import torch
import hydra
import logging
import sys
import os
import numpy as np
import torch.nn as nn
import importlib

from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from torch.serialization import default_restore_location
from lib.train import train
from lib.test import test


def setup_logging():
  ch = logging.StreamHandler(sys.stdout)
  logging.getLogger().setLevel(logging.INFO)
  logging.basicConfig(
      format=os.uname()[1].split('.')[0] + ' %(asctime)s %(message)s',
      datefmt='%m/%d %H:%M:%S',
      handlers=[ch])

# Init datasets and dataloaders 
def my_worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)

def load_state_with_same_shape(model, weights):
    model_state = model.state_dict()
    if list(weights.keys())[0].startswith('module.'):
        print("Loading multigpu weights with module. prefix...")
        weights = {k.partition('module.')[2]:weights[k] for k in weights.keys()}

    if list(weights.keys())[0].startswith('encoder.'):
        logging.info("Loading multigpu weights with encoder. prefix...")
        weights = {k.partition('encoder.')[2]:weights[k] for k in weights.keys()}

    # print(weights.items())
    filtered_weights = {
          k: v for k, v in weights.items() if k in model_state  and v.size() == model_state[k].size()
      }
    print("Loading weights:" + ', '.join(filtered_weights.keys()))
    return filtered_weights

@hydra.main(config_path='config', config_name='default.yaml')
def main(config):
      # load the configurations
    setup_logging()
    if os.path.exists('config.yaml'):
        logging.info('===> Loading exsiting config file')
        config = OmegaConf.load('config.yaml')
        logging.info('===> Loaded exsiting config file')
    logging.info(config.pretty())

    # Create Dataset and Dataloader
    if config.data.dataset == 'sunrgbd':
        from lib.datasets.sunrgbd.sunrgbd_detection_dataset import SunrgbdDetectionVotesDataset, MAX_NUM_OBJ
        from lib.datasets.sunrgbd.model_util_sunrgbd import SunrgbdDatasetConfig
        dataset_config = SunrgbdDatasetConfig()
        train_dataset = SunrgbdDetectionVotesDataset('train', 
            num_points=config.data.num_points,
            augment=True,
            use_color=config.data.use_color, 
            use_height=(not config.data.no_height),
            use_v1=(not config.data.use_sunrgbd_v2),
            data_ratio=config.data.data_ratio)
        test_dataset = SunrgbdDetectionVotesDataset('val', 
            num_points=config.data.num_points,
            augment=False,
            use_color=config.data.use_color, 
            use_height=(not config.data.no_height),
            use_v1=(not config.data.use_sunrgbd_v2))
    elif config.data.dataset == 'scannet':
        from lib.datasets.scannet.scannet_detection_dataset import ScannetDetectionDataset, MAX_NUM_OBJ
        from lib.datasets.scannet.model_util_scannet import ScannetDatasetConfig
        dataset_config = ScannetDatasetConfig()
        train_dataset = ScannetDetectionDataset('train', 
            num_points=config.data.num_points,
            augment=True,
            use_color=config.data.use_color, 
            use_height=(not config.data.no_height),
            data_ratio=config.data.data_ratio)
        test_dataset = ScannetDetectionDataset('val', 
            num_points=config.data.num_points,
            augment=False,
            use_color=config.data.use_color, 
            use_height=(not config.data.no_height))
    else:
        logging.info('Unknown dataset %s. Exiting...'%(config.data.dataset))
        exit(-1)

    COLLATE_FN = None
    if config.data.voxelization:
        from models.backbone.sparseconv.voxelized_dataset import VoxelizationDataset, collate_fn
        train_dataset = VoxelizationDataset(train_dataset, config.data.voxel_size)
        test_dataset = VoxelizationDataset(test_dataset, config.data.voxel_size)
        COLLATE_FN = collate_fn

    logging.info('training: {}, testing: {}'.format(len(train_dataset), len(test_dataset)))

    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=config.data.batch_size,
        shuffle=True, 
        num_workers=config.data.num_workers, 
        worker_init_fn=my_worker_init_fn,
        collate_fn=COLLATE_FN)

    test_dataloader = DataLoader(
        test_dataset, 
        batch_size=config.data.num_workers,
        shuffle=True, 
        num_workers=config.data.num_workers, 
        worker_init_fn=my_worker_init_fn,
        collate_fn=COLLATE_FN)

    logging.info('train dataloader: {}, test dataloader: {}'.format(len(train_dataloader),len(test_dataloader)))

    # Init the model and optimzier
    MODEL = importlib.import_module('models.' + config.net.model) # import network module
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    num_input_channel = int(config.data.use_color)*3 + int(not config.data.no_height)*1

    if config.net.model == 'boxnet':
        Detector = MODEL.BoxNet
    else:
        Detector = MODEL.VoteNet

    net = Detector(num_class=dataset_config.num_class,
                   num_heading_bin=dataset_config.num_heading_bin,
                   num_size_cluster=dataset_config.num_size_cluster,
                   mean_size_arr=dataset_config.mean_size_arr,
                   num_proposal=config.net.num_target,
                   input_feature_dim=num_input_channel,
                   vote_factor=config.net.vote_factor,
                   sampling=config.net.cluster_sampling,
                   backbone=config.net.backbone)

    if config.net.weights is not None:
        assert config.net.backbone == "sparseconv", "only support sparseconv"
        print('===> Loading weights: ' + config.net.weights)
        state = torch.load(config.net.weights, map_location=lambda s, l: default_restore_location(s, 'cpu'))
        model = net
        if config.net.is_train:
            model = net.backbone_net.net
        matched_weights = load_state_with_same_shape(model, state['state_dict'])
        model_dict = model.state_dict()
        model_dict.update(matched_weights)
        model.load_state_dict(model_dict)

        # from pdb import set_trace; set_trace()

    net.to(device)

    if config.net.is_train:
        train(net, train_dataloader, test_dataloader, dataset_config, config)
    else:
        test(net, test_dataloader, dataset_config, config)

if __name__ == "__main__":
    main()

