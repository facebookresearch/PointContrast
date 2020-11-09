# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

""" Evaluation routine for 3D object detection with SUN RGB-D and ScanNet.
"""

import os
import sys
import logging
import numpy as np
from datetime import datetime
import argparse
import importlib
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from models.ap_helper import APCalculator, parse_predictions, parse_groundtruths
from models.dump_helper import dump_results
from models.loss_helper import get_loss as criterion

def test(net, test_dataloader, dataset_config, config):
    if config.test.use_cls_nms:
        assert(config.test.use_3d_nms)
    # Used for AP calculation
    CONFIG_DICT = {'remove_empty_box': (not config.test.faster_eval), 
                   'use_3d_nms': config.test.use_3d_nms, 
                   'nms_iou': config.test.nms_iou,
                   'use_old_type_nms': config.test.use_old_type_nms, 
                   'cls_nms': config.test.use_cls_nms, 
                   'per_class_proposal': config.test.per_class_proposal,
                   'conf_thresh': config.test.conf_thresh, 
                   'dataset_config': dataset_config}

    AP_IOU_THRESHOLDS = config.test.ap_iou_thresholds
    logging.info(str(datetime.now()))
    # Reset numpy seed.
    # REF: https://github.com/pytorch/pytorch/issues/5059
    np.random.seed()

    stat_dict = {}
    ap_calculator_list = [APCalculator(iou_thresh, CONFIG_DICT['dataset_config'].class2type) \
        for iou_thresh in AP_IOU_THRESHOLDS]
    net.eval() # set model to eval mode (for bn and dp)
    for batch_idx, batch_data_label in enumerate(test_dataloader):
        if batch_idx % 10 == 0:
            print('Eval batch: %d'%(batch_idx))
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
        for ap_calculator in ap_calculator_list:
            ap_calculator.step(batch_pred_map_cls, batch_gt_map_cls)
    
        # Dump evaluation results for visualization
        if batch_idx == 0:
            dump_results(end_points, 'visualization', CONFIG_DICT['dataset_config'])

    # Log statistics
    for key in sorted(stat_dict.keys()):
        logging.info('eval mean %s: %f'%(key, stat_dict[key]/(float(batch_idx+1))))

    # Evaluate average precision
    for i, ap_calculator in enumerate(ap_calculator_list):
        logging.info('-'*10 + 'iou_thresh: %f'%(AP_IOU_THRESHOLDS[i]) + '-'*10)
        metrics_dict = ap_calculator.compute_metrics()
        for key in metrics_dict:
            logging.info('eval %s: %f'%(key, metrics_dict[key]))

    mean_loss = stat_dict['loss']/float(batch_idx+1)
    return mean_loss

