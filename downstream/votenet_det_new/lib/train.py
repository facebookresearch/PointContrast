# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

""" Training routine for 3D object detection with SUN RGB-D or ScanNet.

Sample usage:
python train.py --dataset sunrgbd --log_dir log_sunrgbd

To use Tensorboard:
At server:
    python -m tensorboard.main --logdir=<log_dir_name> --port=6006
At local machine:
    ssh -L 1237:localhost:6006 <server_name>
Then go to local browser and type:
    localhost:1237
"""

import os
import sys
import numpy as np
from datetime import datetime
import argparse
import importlib
import logging
from omegaconf import OmegaConf

from models.loss_helper import get_loss as criterion
from tensorboardX import SummaryWriter

import torch
import torch.optim as optim
from torch.optim import lr_scheduler

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from models.backbone.pointnet2.pytorch_utils import BNMomentumScheduler
from models.dump_helper import dump_results
from models.ap_helper import APCalculator, parse_predictions, parse_groundtruths


def get_current_lr(epoch, config):

    lr = config.optimizer.learning_rate
    for i,lr_decay_epoch in enumerate(config.optimizer.lr_decay_steps):
        if epoch >= lr_decay_epoch:
            lr *= config.optimizer.lr_decay_rates[i]
    return lr

def adjust_learning_rate(optimizer, epoch, config):
    lr = get_current_lr(epoch, config)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def train_one_epoch(net, train_dataloader, optimizer, bnm_scheduler, epoch_cnt, dataset_config, writer, config):
    stat_dict = {} # collect statistics
    adjust_learning_rate(optimizer, epoch_cnt, config)
    bnm_scheduler.step() # decay BN momentum
    net.train() # set model to training mode
    for batch_idx, batch_data_label in enumerate(train_dataloader):
        for key in batch_data_label:
            batch_data_label[key] = batch_data_label[key].cuda()

        # Forward pass
        optimizer.zero_grad()
        inputs = {'point_clouds': batch_data_label['point_clouds']}
        if 'voxel_coords' in batch_data_label:
            inputs.update({
                'voxel_coords': batch_data_label['voxel_coords'],
                'voxel_inds':   batch_data_label['voxel_inds'],
                'voxel_feats':  batch_data_label['voxel_feats']})

        end_points = net(inputs)
        
        # Compute loss and gradients, update parameters.
        for key in batch_data_label:
            assert(key not in end_points)
            end_points[key] = batch_data_label[key]
        loss, end_points = criterion(end_points, dataset_config)
        loss.backward()
        optimizer.step()

        # Accumulate statistics and print out
        for key in end_points:
            if 'loss' in key or 'acc' in key or 'ratio' in key:
                if key not in stat_dict: stat_dict[key] = 0
                stat_dict[key] += end_points[key].item()

        batch_interval = 10
        if (batch_idx+1) % batch_interval == 0:
            logging.info(' ---- batch: %03d ----' % (batch_idx+1))
            for key in stat_dict:
                writer.add_scalar('training/{}'.format(key), stat_dict[key]/batch_interval, 
                                  (epoch_cnt*len(train_dataloader)+batch_idx)*config.data.batch_size)
            for key in sorted(stat_dict.keys()):
                logging.info('mean %s: %f'%(key, stat_dict[key]/batch_interval))
                stat_dict[key] = 0

def evaluate_one_epoch(net, train_dataloader, test_dataloader, config, epoch_cnt, CONFIG_DICT, writer):
    stat_dict = {} # collect statistics
    ap_calculator = APCalculator(ap_iou_thresh=0.5, class2type_map=CONFIG_DICT['dataset_config'].class2type)
    net.eval() # set model to eval mode (for bn and dp)
    for batch_idx, batch_data_label in enumerate(test_dataloader):
        if batch_idx % 10 == 0:
            logging.info('Eval batch: %d'%(batch_idx))
        for key in batch_data_label:
            batch_data_label[key] = batch_data_label[key].cuda()
        
        # Forward pass
        inputs = {'point_clouds': batch_data_label['point_clouds']}
        if 'voxel_coords' in batch_data_label:
            inputs.update({
                'voxel_coords': batch_data_label['voxel_coords'],
                'voxel_inds':   batch_data_label['voxel_inds'],
                'voxel_feats':  batch_data_label['voxel_feats']})

        with torch.no_grad():
            end_points = net(inputs)

        # Compute loss
        for key in batch_data_label:
            assert(key not in end_points)
            end_points[key] = batch_data_label[key]
        loss, end_points = criterion(end_points, CONFIG_DICT['dataset_config'])

        # Accumulate statistics and print out
        for key in end_points:
            if 'loss' in key or 'acc' in key or 'ratio' in key:
                if key not in stat_dict: stat_dict[key] = 0
                stat_dict[key] += end_points[key].item()

        batch_pred_map_cls = parse_predictions(end_points, CONFIG_DICT) 
        batch_gt_map_cls = parse_groundtruths(end_points, CONFIG_DICT) 
        ap_calculator.step(batch_pred_map_cls, batch_gt_map_cls)

        # Dump evaluation results for visualization
        if config.data.dump_results and batch_idx == 0 and epoch_cnt %10 == 0:
            dump_results(end_points, 'results', CONFIG_DICT['dataset_config']) 

    # Log statistics
    for key in sorted(stat_dict.keys()):
        writer.add_scalar('validation/{}'.format(key), stat_dict[key]/float(batch_idx+1),
                          (epoch_cnt+1)*len(train_dataloader)*config.data.batch_size)
        logging.info('eval mean %s: %f'%(key, stat_dict[key]/(float(batch_idx+1))))

    # Evaluate average precision
    metrics_dict = ap_calculator.compute_metrics()
    for key in metrics_dict:
        logging.info('eval %s: %f'%(key, metrics_dict[key]))
    writer.add_scalar('validation/mAP@0.5', metrics_dict['mAP'], (epoch_cnt+1)*len(train_dataloader)*config.data.batch_size)

    mean_loss = stat_dict['loss']/float(batch_idx+1)
    return mean_loss


def train(net, train_dataloader, test_dataloader, dataset_config, config):

    # Used for AP calculation
    CONFIG_DICT = {'remove_empty_box':False, 'use_3d_nms':True,
        'nms_iou':0.25, 'use_old_type_nms':False, 'cls_nms':True,
        'per_class_proposal': True, 'conf_thresh':0.05,
        'dataset_config': dataset_config}

    # Load the Adam optimizer
    optimizer = optim.Adam(net.parameters(), lr=config.optimizer.learning_rate, weight_decay=config.optimizer.weight_decay)

    # writer
    writer = SummaryWriter(log_dir='tensorboard')

    # Load checkpoint if there is any
    start_epoch = 0
    CHECKPOINT_PATH = os.path.join('checkpoint.tar')
    if os.path.isfile(CHECKPOINT_PATH):
        checkpoint = torch.load(CHECKPOINT_PATH)
        net.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        logging.info("-> loaded checkpoint %s (epoch: %d)"%(CHECKPOINT_PATH, start_epoch))

    # Decay Batchnorm momentum from 0.5 to 0.999
    # note: pytorch's BN momentum (default 0.1)= 1 - tensorflow's BN momentum
    BN_MOMENTUM_INIT = 0.5
    BN_MOMENTUM_MAX = 0.001
    BN_DECAY_STEP = config.optimizer.bn_decay_step
    BN_DECAY_RATE = config.optimizer.bn_decay_rate
    bn_lbmd = lambda it: max(BN_MOMENTUM_INIT * BN_DECAY_RATE**(int(it / BN_DECAY_STEP)), BN_MOMENTUM_MAX)
    bnm_scheduler = BNMomentumScheduler(net, bn_lambda=bn_lbmd, last_epoch=start_epoch-1)

    loss = 0
    for epoch in range(start_epoch, config.optimizer.max_epoch):
        logging.info('**** EPOCH %03d ****' % (epoch))
        logging.info('Current learning rate: %f'%(get_current_lr(epoch, config)))
        logging.info('Current BN decay momentum: %f'%(bnm_scheduler.lmbd(bnm_scheduler.last_epoch)))
        logging.info(str(datetime.now()))
        # Reset numpy seed.
        # REF: https://github.com/pytorch/pytorch/issues/5059
        np.random.seed()
        train_one_epoch(net=net, train_dataloader=train_dataloader, optimizer=optimizer, 
                        bnm_scheduler=bnm_scheduler, epoch_cnt=epoch, dataset_config=dataset_config, 
                        writer=writer, config=config)
        if epoch == 0 or epoch % 5 == 4: # Eval every 5 epochs
            loss = evaluate_one_epoch(net=net, train_dataloader=train_dataloader, test_dataloader=test_dataloader, 
                                      config=config, epoch_cnt=epoch, CONFIG_DICT=CONFIG_DICT, writer=writer)
        # Save checkpoint
        save_dict = {'epoch': epoch+1, # after training one epoch, the start_epoch should be epoch+1
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss,
                    }
        try: # with nn.DataParallel() the net is added as a submodule of DataParallel
            save_dict['state_dict'] = net.module.state_dict()
        except:
            save_dict['state_dict'] = net.state_dict()
        torch.save(save_dict, 'checkpoint.tar')
        OmegaConf.save(config, 'config.yaml')

