# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import json
import logging
import os
import errno
import time

import numpy as np
from omegaconf import OmegaConf
import torch

from lib.pc_utils import colorize_pointcloud, save_point_cloud
from lib.distributed_utils import get_world_size, get_rank

def load_state_with_same_shape(model, weights):
  print(weights.keys())
  # weights['conv1.kernel'] = weights['conv1.kernel'].repeat([1,3,1])/3.0
  model_state = model.state_dict()
  if list(weights.keys())[0].startswith('module.'):
    logging.info("Loading multigpu weights with module. prefix...")
    weights = {k.partition('module.')[2]:weights[k] for k in weights.keys()}
  
  if list(weights.keys())[0].startswith('encoder.'):
    logging.info("Loading multigpu weights with encoder. prefix...")
    weights = {k.partition('encoder.')[2]:weights[k] for k in weights.keys()}
  
  # print(weights.items())
  # print("===================")
  # print("===================")
  # print("===================")
  # print("===================")
  # print("===================")
  # print(model_state)
  
  filtered_weights = {
      k: v for k, v in weights.items() if k in model_state  and v.size() == model_state[k].size() 
  }
  logging.info("Loading weights:" + ', '.join(filtered_weights.keys()))
  return filtered_weights

def load_state_with_same_shape_no_conv0(model, weights):
  model_state = model.state_dict()
  filtered_weights = {
      k: v for k, v in weights.items() if k in model_state and v.size() == model_state[k].size() and "conv0p1s1" not in k
  }
  logging.info("Loading weights:" + ', '.join(filtered_weights.keys()))
  return filtered_weights

def load_state_with_same_shape_no_bn0(model, weights):
  model_state = model.state_dict()
  filtered_weights = {
      k: v for k, v in weights.items() if k in model_state and v.size() == model_state[k].size() and "bn0" not in k
  }
  logging.info("Loading weights:" + ', '.join(filtered_weights.keys()))
  return filtered_weights

def load_state_with_same_shape_no_bn(model, weights):
  model_state = model.state_dict()
  filtered_weights = {
      k: v for k, v in weights.items() if k in model_state and v.size() == model_state[k].size() and "norm" not in k and "bn" not in k and "running" not in k and "tracked" not in k
  }
  logging.info("Loading weights:" + ', '.join(filtered_weights.keys()))
  return filtered_weights

def load_state_with_same_shape_no_bn_stats(model, weights):
  model_state = model.state_dict()
  filtered_weights = {
      k: v for k, v in weights.items() if k in model_state and v.size() == model_state[k].size() and "running" not in k and "tracked" not in k

  }
  logging.info("Loading weights:" + ', '.join(filtered_weights.keys()))
  return filtered_weights

def checkpoint(model, optimizer, epoch, iteration, config, best_val=None, best_val_iter=None, postfix=None):
  # only works for rank == 0
  if get_world_size() > 1 and get_rank() > 0:
    return

  mkdir_p('weights')
  if config.train.overwrite_weights:
    if postfix is not None:
      filename = f"checkpoint_{config.net.wrapper_type}{config.net.model}{postfix}.pth"
    else:
      filename = f"checkpoint_{config.net.wrapper_type}{config.net.model}.pth"
  else:
    filename = f"checkpoint_{config.net.wrapper_type}{config.net.model}_iter_{iteration}.pth"
  checkpoint_file = 'weights/' + filename

  _model = model.module if get_world_size() > 1 else model
  state = {
      'iteration': iteration,
      'epoch': epoch,
      'arch': config.net.model,
      'state_dict': _model.state_dict(),
      'optimizer': optimizer.state_dict()
  }
  if best_val is not None:
    state['best_val'] = best_val
    state['best_val_iter'] = best_val_iter

  # save config
  OmegaConf.save(config, 'config.yaml')

  torch.save(state, checkpoint_file)
  logging.info(f"Checkpoint saved to {checkpoint_file}")
  # Delete symlink if it exists
  if os.path.exists('weights/weights.pth'):
    os.remove('weights/weights.pth')
  # Create symlink
  os.system('ln -s {} weights/weights.pth'.format(filename))


def precision_at_one(pred, target, ignore_label=255):
  """Computes the precision@k for the specified values of k"""
  # batch_size = target.size(0) * target.size(1) * target.size(2)
  pred = pred.view(1, -1)
  target = target.view(1, -1)
  correct = pred.eq(target)
  correct = correct[target != ignore_label]
  correct = correct.view(-1)
  if correct.nelement():
    return correct.float().sum(0).mul(100.0 / correct.size(0)).item()
  else:
    return float('nan')


def fast_hist(pred, label, n):
  k = (label >= 0) & (label < n)
  return np.bincount(n * label[k].astype(int) + pred[k], minlength=n**2).reshape(n, n)


def per_class_iu(hist):
  with np.errstate(divide='ignore', invalid='ignore'):
    return np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))


class WithTimer(object):
  """Timer for with statement."""

  def __init__(self, name=None):
    self.name = name

  def __enter__(self):
    self.tstart = time.time()

  def __exit__(self, type, value, traceback):
    out_str = 'Elapsed: %s' % (time.time() - self.tstart)
    if self.name:
      logging.info('[{self.name}]')
    logging.info(out_str)


class Timer(object):
  """A simple timer."""

  def __init__(self):
    self.total_time = 0.
    self.calls = 0
    self.start_time = 0.
    self.diff = 0.
    self.average_time = 0.

  def reset(self):
    self.total_time = 0
    self.calls = 0
    self.start_time = 0
    self.diff = 0
    self.averate_time = 0

  def tic(self):
    # using time.time instead of time.clock because time time.clock
    # does not normalize for multithreading
    self.start_time = time.time()

  def toc(self, average=True):
    self.diff = time.time() - self.start_time
    self.total_time += self.diff
    self.calls += 1
    self.average_time = self.total_time / self.calls
    if average:
      return self.average_time
    else:
      return self.diff


class ExpTimer(Timer):
  """ Exponential Moving Average Timer """

  def __init__(self, alpha=0.5):
    super(ExpTimer, self).__init__()
    self.alpha = alpha

  def toc(self):
    self.diff = time.time() - self.start_time
    self.average_time = self.alpha * self.diff + \
        (1 - self.alpha) * self.average_time
    return self.average_time


class AverageMeter(object):
  """Computes and stores the average and current value"""

  def __init__(self):
    self.reset()

  def reset(self):
    self.val = 0
    self.avg = 0
    self.sum = 0
    self.count = 0

  def update(self, val, n=1):
    self.val = val
    self.sum += val * n
    self.count += n
    self.avg = self.sum / self.count


def mkdir_p(path):
  try:
    os.makedirs(path)
  except OSError as exc:
    if exc.errno == errno.EEXIST and os.path.isdir(path):
      pass
    else:
      raise


def read_txt(path):
  """Read txt file into lines.
  """
  with open(path) as f:
    lines = f.readlines()
  lines = [x.strip() for x in lines]
  return lines


def debug_on():
  import sys
  import pdb
  import functools
  import traceback

  def decorator(f):

    @functools.wraps(f)
    def wrapper(*args, **kwargs):
      try:
        return f(*args, **kwargs)
      except Exception:
        info = sys.exc_info()
        traceback.print_exception(*info)
        pdb.post_mortem(info[2])

    return wrapper

  return decorator


def get_prediction(dataset, output, target):
  return output.max(1)[1]


def count_parameters(model):
  return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_torch_device(is_cuda):
  return torch.device('cuda' if is_cuda else 'cpu')


class HashTimeBatch(object):

  def __init__(self, prime=5279):
    self.prime = prime

  def __call__(self, time, batch):
    return self.hash(time, batch)

  def hash(self, time, batch):
    return self.prime * batch + time

  def dehash(self, key):
    time = key % self.prime
    batch = key / self.prime
    return time, batch


def save_rotation_pred(iteration, pred, dataset, save_pred_dir):
  """Save prediction results in original pointcloud scale."""
  decode_label_map = {}
  for k, v in dataset.label_map.items():
    decode_label_map[v] = k
  pred = np.array([decode_label_map[x] for x in pred], dtype=np.int)
  out_rotation_txt = dataset.get_output_id(iteration) + '.txt'
  out_rotation_path = save_pred_dir + '/' + out_rotation_txt
  np.savetxt(out_rotation_path, pred, fmt='%i')


def save_predictions(coords, upsampled_pred, transformation, dataset, config, iteration,
                     save_pred_dir):
  """Save prediction results in original pointcloud scale."""
  from lib.dataset import OnlineVoxelizationDatasetBase
  if dataset.IS_ONLINE_VOXELIZATION:
    assert transformation is not None, 'Need transformation matrix.'
  iter_size = coords[:, -1].max() + 1  # Normally batch_size, may be smaller at the end.
  if dataset.IS_TEMPORAL:  # Iterate over temporal dilation.
    iter_size *= config.temporal_numseq
  for i in range(iter_size):
    # Get current pointcloud filtering mask.
    if dataset.IS_TEMPORAL:
      j = i % config.temporal_numseq
      i = i // config.temporal_numseq
    batch_mask = coords[:, -1].numpy() == i
    if dataset.IS_TEMPORAL:
      batch_mask = np.logical_and(batch_mask, coords[:, -2].numpy() == j)
    # Calculate original coordinates.
    coords_original = coords[:, :3].numpy()[batch_mask] + 0.5
    if dataset.IS_ONLINE_VOXELIZATION:
      # Undo voxelizer transformation.
      curr_transformation = transformation[i, :16].numpy().reshape(4, 4)
      xyz = np.hstack((coords_original, np.ones((batch_mask.sum(), 1))))
      orig_coords = (np.linalg.inv(curr_transformation) @ xyz.T).T
    else:
      orig_coords = coords_original
    orig_pred = upsampled_pred[batch_mask]
    # Undo ignore label masking to fit original dataset label.
    if dataset.IGNORE_LABELS:
      if isinstance(dataset, OnlineVoxelizationDatasetBase):
        label2masked = dataset.label2masked
        maskedmax = label2masked[label2masked < 255].max() + 1
        masked2label = [label2masked.tolist().index(i) for i in range(maskedmax)]
        orig_pred = np.take(masked2label, orig_pred)
      else:
        decode_label_map = {}
        for k, v in dataset.label_map.items():
          decode_label_map[v] = k
        orig_pred = np.array([decode_label_map[x] for x in orig_pred], dtype=np.int)
    # Determine full path of the destination.
    full_pred = np.hstack((orig_coords[:, :3], np.expand_dims(orig_pred, 1)))
    filename = 'pred_%04d_%02d.npy' % (iteration, i)
    if dataset.IS_TEMPORAL:
      filename = 'pred_%04d_%02d_%02d.npy' % (iteration, i, j)
    # Save final prediction as npy format.
    np.save(os.path.join(save_pred_dir, filename), full_pred)


def visualize_results(coords, input, target, upsampled_pred, config, iteration):
  # Get filter for valid predictions in the first batch.
  target_batch = coords[:, 3].numpy() == 0
  input_xyz = coords[:, :3].numpy()
  target_valid = target.numpy() != 255
  target_pred = np.logical_and(target_batch, target_valid)
  target_nonpred = np.logical_and(target_batch, ~target_valid)
  ptc_nonpred = np.hstack((input_xyz[target_nonpred], np.zeros((np.sum(target_nonpred), 3))))
  # Unwrap file index if tested with rotation.
  file_iter = iteration
  if config.test_rotation >= 1:
    file_iter = iteration // config.test.test_rotation
  # Create directory to save visualization results.
  os.makedirs(config.test.visualize_path, exist_ok=True)
  # Label visualization in RGB.
  xyzlabel = colorize_pointcloud(input_xyz[target_pred], upsampled_pred[target_pred])
  xyzlabel = np.vstack((xyzlabel, ptc_nonpred))
  filename = '_'.join([config.dataset, config.model, 'pred', '%04d.ply' % file_iter])
  save_point_cloud(xyzlabel, os.path.join(config.test.visualize_path, filename), verbose=False)
  # RGB input values visualization.
  xyzrgb = np.hstack((input_xyz[target_batch], input[:, :3].cpu().numpy()[target_batch]))
  filename = '_'.join([config.dataset, config.model, 'rgb', '%04d.ply' % file_iter])
  save_point_cloud(xyzrgb, os.path.join(config.test.visualize_path, filename), verbose=False)
  # Ground-truth visualization in RGB.
  xyzgt = colorize_pointcloud(input_xyz[target_pred], target.numpy()[target_pred])
  xyzgt = np.vstack((xyzgt, ptc_nonpred))
  filename = '_'.join([config.dataset, config.model, 'gt', '%04d.ply' % file_iter])
  save_point_cloud(xyzgt, os.path.join(config.test.visualize_path, filename), verbose=False)


def permute_pointcloud(input_coords, pointcloud, transformation, label_map,
                       voxel_output, voxel_pred):
  """Get permutation from pointcloud to input voxel coords."""
  def _hash_coords(coords, coords_min, coords_dim):
    return np.ravel_multi_index((coords - coords_min).T, coords_dim)
  # Validate input.
  input_batch_size = input_coords[:, -1].max().item()
  pointcloud_batch_size = pointcloud[:, -1].max().int().item()
  transformation_batch_size = transformation[:, -1].max().int().item()
  assert input_batch_size == pointcloud_batch_size == transformation_batch_size
  pointcloud_permutation, pointcloud_target = [], []
  # Process each batch.
  for i in range(input_batch_size + 1):
    # Filter batch from the data.
    input_coords_mask_b = input_coords[:, -1] == i
    input_coords_b = (input_coords[input_coords_mask_b])[:, :-1].numpy()
    pointcloud_b = pointcloud[pointcloud[:, -1] == i, :-1].numpy()
    transformation_b = transformation[i, :-1].reshape(4, 4).numpy()
    # Transform original pointcloud to voxel space.
    original_coords1 = np.hstack((pointcloud_b[:, :3], np.ones((pointcloud_b.shape[0], 1))))
    original_vcoords = np.floor(original_coords1 @ transformation_b.T)[:, :3].astype(int)
    # Hash input and voxel coordinates to flat coordinate.
    vcoords_all = np.vstack((input_coords_b, original_vcoords))
    vcoords_min = vcoords_all.min(0)
    vcoords_dims = vcoords_all.max(0) - vcoords_all.min(0) + 1
    input_coords_key = _hash_coords(input_coords_b, vcoords_min, vcoords_dims)
    original_vcoords_key = _hash_coords(original_vcoords, vcoords_min, vcoords_dims)
    # Query voxel predictions from original pointcloud.
    key_to_idx = dict(zip(input_coords_key, range(len(input_coords_key))))
    pointcloud_permutation.append(
        np.array([key_to_idx.get(i, -1) for i in original_vcoords_key]))
    pointcloud_target.append(pointcloud_b[:, -1].astype(int))
  pointcloud_permutation = np.concatenate(pointcloud_permutation)
  # Prepare pointcloud permutation array.
  pointcloud_permutation = torch.from_numpy(pointcloud_permutation)
  permutation_mask = pointcloud_permutation >= 0
  permutation_valid = pointcloud_permutation[permutation_mask]
  # Permuate voxel output to pointcloud.
  pointcloud_output = torch.zeros(pointcloud.shape[0], voxel_output.shape[1]).to(voxel_output)
  pointcloud_output[permutation_mask] = voxel_output[permutation_valid]
  # Permuate voxel prediction to pointcloud.
  # NOTE: Invalid points (points found in pointcloud but not in the voxel) are mapped to 0.
  pointcloud_pred = torch.ones(pointcloud.shape[0]).int().to(voxel_pred) * 0
  pointcloud_pred[permutation_mask] = voxel_pred[permutation_valid]
  # Map pointcloud target to respect dataset IGNORE_LABELS
  pointcloud_target = torch.from_numpy(
      np.array([label_map[i] for i in np.concatenate(pointcloud_target)])).int()
  return pointcloud_output, pointcloud_pred, pointcloud_target
