import os
import os.path as osp
import gc
import logging
import numpy as np
import json
from omegaconf import OmegaConf
import torch.nn as nn

import torch
import torch.optim as optim
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import RandomSampler
from lib.dataloader import InfSampler, DistributedInfSampler

from model import load_model
import util.transform_estimation as te
from lib.metrics import pdist, corr_dist
from lib.timer import Timer, AverageMeter
from lib.eval import find_nn_gpu

from util.file import ensure_dir
from util.misc import _hash

import MinkowskiEngine as ME

import lib.multiprocessing as mpu
import lib.distributed as du
import torch.distributed as dist

from lib.criterion import NCESoftmaxLoss

from torch.serialization import default_restore_location

torch.autograd.set_detect_anomaly(True)

def shuffle_loader(data_loader, cur_epoch):
  assert isinstance(data_loader.sampler, (RandomSampler, InfSampler, DistributedSampler, DistributedInfSampler))
  if isinstance(data_loader.sampler, DistributedSampler):
    data_loader.sampler.set_epoch(cur_epoch)

# TODO(s9xie): move to utils
def load_state_with_same_shape(model, weights):
  model_state = model.state_dict()
  filtered_weights = {
      k: v for k, v in weights.items() if k in model_state and v.size() == model_state[k].size()
  }
  logging.info("Loading weights:" + ', '.join(filtered_weights.keys()))
  return filtered_weights


class ContrastiveLossTrainer:
  def __init__(
      self,
      config,
      data_loader):
    assert config.misc.use_gpu and torch.cuda.is_available(), "DDP mode must support GPU"
    num_feats = 3  # always 3 for finetuning.

    self.is_master = du.is_master_proc(config.misc.num_gpus) if config.misc.num_gpus > 1 else True

    # Model initialization
    self.cur_device = torch.cuda.current_device()
    Model = load_model(config.net.model)
    model = Model(
        num_feats,
        config.net.model_n_out,
        config,
        D=3)
    model = model.cuda(device=self.cur_device)
    if config.misc.num_gpus > 1:
        model = torch.nn.parallel.DistributedDataParallel(
                module=model,
                device_ids=[self.cur_device],
                output_device=self.cur_device,
                broadcast_buffers=False,
        )
    if config.misc.weights:
      checkpoint = torch.load(config.misc.weights, map_location=lambda s, l: default_restore_location(s, 'cpu'))
      model.load_state_dict(checkpoint['state_dict'])

    self.config = config
    self.model = model
    self.max_epoch = config.opt.max_epoch
    self.save_freq = config.trainer.save_freq_epoch

    self.optimizer = getattr(optim, config.opt.optimizer)(
        model.parameters(),
        lr=config.opt.lr,
        momentum=config.opt.momentum,
        weight_decay=config.opt.weight_decay)

    self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, config.opt.exp_gamma)

    self.start_epoch = 1
    self.checkpoint_dir = config.logging.out_dir

    if self.is_master:
        ensure_dir(self.checkpoint_dir)
        OmegaConf.save(config, os.path.join(self.checkpoint_dir, 'config.yaml'))

    self.iter_size = config.opt.iter_size
    self.batch_size = data_loader.batch_size
    self.data_loader = data_loader

    self.log_step = int(np.sqrt(self.batch_size))
    self.writer = SummaryWriter(logdir=config.logging.out_dir)

    self.neg_thresh = config.trainer.neg_thresh
    self.pos_thresh = config.trainer.pos_thresh
    self.neg_weight = config.trainer.neg_weight

    if config.misc.resume is not None:
      if osp.isfile(config.misc.resume):
        logging.info("=> loading checkpoint '{}'".format(config.misc.resume))
        state = torch.load(config.misc.resume, map_location=lambda s, l: default_restore_location(s, 'cpu'))
        self.start_epoch = state['epoch']

        if config.misc.lenient_weight_loading:
          matched_weights = load_state_with_same_shape(model, state['state_dict'])
          model_dict = model.state_dict()
          model_dict.update(matched_weights)
          model.load_state_dict(model_dict)
        else:
          model.load_state_dict(state['state_dict'])
          self.scheduler.load_state_dict(state['scheduler'])
          self.optimizer.load_state_dict(state['optimizer'])
      else:
        raise ValueError(f"=> no checkpoint found at '{config.misc.resume}'")

  def _save_checkpoint(self, epoch, filename='checkpoint'):
    if not self.is_master:
        return
    state = {
        'epoch': epoch,
        'state_dict': self.model.state_dict(),
        'optimizer': self.optimizer.state_dict(),
        'scheduler': self.scheduler.state_dict(),
        'config': self.config
    }
    filename = os.path.join(self.checkpoint_dir, f'{filename}.pth')
    logging.info("Saving checkpoint: {} ...".format(filename))
    torch.save(state, filename)

class HardestContrastiveLossTrainer(ContrastiveLossTrainer):

  def __init__(
      self,
      config,
      data_loader):
    ContrastiveLossTrainer.__init__(self, config, data_loader)
 
    self.stat_freq = config.trainer.stat_freq
    self.lr_update_freq = config.trainer.lr_update_freq

  def contrastive_hardest_negative_loss(self,
                                        F0,
                                        F1,
                                        positive_pairs,
                                        num_pos=5192,
                                        num_hn_samples=2048,
                                        thresh=None):
    """
    Generate negative pairs
    """
    N0, N1 = len(F0), len(F1)
    N_pos_pairs = len(positive_pairs)
    hash_seed = max(N0, N1)
    sel0 = np.random.choice(N0, min(N0, num_hn_samples), replace=False)
    sel1 = np.random.choice(N1, min(N1, num_hn_samples), replace=False)

    if N_pos_pairs > num_pos:
      pos_sel = np.random.choice(N_pos_pairs, num_pos, replace=False)
      sample_pos_pairs = positive_pairs[pos_sel]
    else:
      sample_pos_pairs = positive_pairs

    # Find negatives for all F1[positive_pairs[:, 1]]
    subF0, subF1 = F0[sel0], F1[sel1]

    pos_ind0 = sample_pos_pairs[:, 0].long()
    pos_ind1 = sample_pos_pairs[:, 1].long()
    posF0, posF1 = F0[pos_ind0], F1[pos_ind1]
    
    D01 = pdist(posF0, subF1, dist_type='L2')
    D10 = pdist(posF1, subF0, dist_type='L2')

    D01min, D01ind = D01.min(1)
    D10min, D10ind = D10.min(1)

    if not isinstance(positive_pairs, np.ndarray):
      positive_pairs = np.array(positive_pairs, dtype=np.int64)

    pos_keys = _hash(positive_pairs, hash_seed)

    D01ind = sel1[D01ind.cpu().numpy()]
    D10ind = sel0[D10ind.cpu().numpy()]
    neg_keys0 = _hash([pos_ind0.numpy(), D01ind], hash_seed)
    neg_keys1 = _hash([D10ind, pos_ind1.numpy()], hash_seed)

    mask0 = torch.from_numpy(
        np.logical_not(np.isin(neg_keys0, pos_keys, assume_unique=False)))
    mask1 = torch.from_numpy(
        np.logical_not(np.isin(neg_keys1, pos_keys, assume_unique=False)))
    pos_loss = F.relu((posF0 - posF1).pow(2).sum(1) - self.pos_thresh)
    neg_loss0 = F.relu(self.neg_thresh - D01min[mask0]).pow(2)
    neg_loss1 = F.relu(self.neg_thresh - D10min[mask1]).pow(2)
    return pos_loss.mean(), (neg_loss0.mean() + neg_loss1.mean()) / 2

  def train(self):

    assert self.config.trainer.infinite_sampler, "Only support InfSampler"
    
    curr_iter, epoch = 0, 0
    data_loader = self.data_loader
    data_loader_iter = self.data_loader.__iter__()
    iter_size = self.iter_size
    start_iter = 0
    data_meter, data_timer, total_timer = AverageMeter(), Timer(), Timer()
    
    total_loss = 0
    total_num = 0.0

    while (curr_iter < self.config.opt.max_iter):

      curr_iter += 1
      epoch = curr_iter / len(self.data_loader)

      if curr_iter % self.lr_update_freq == 0 or curr_iter == 1:
        lr = self.scheduler.get_lr()
        if self.is_master:
          logging.info(f" Epoch: {epoch}, LR: {lr}")
          self._save_checkpoint(epoch, 'checkpoint_'+str(curr_iter))
          self.scheduler.step()

      batch_loss, batch_pos_loss, batch_neg_loss = self._train_iter(curr_iter, data_loader_iter, start_iter, iter_size, [data_meter, data_timer, total_timer])

      total_loss += batch_loss
      total_num += 1

      if curr_iter % self.config.trainer.stat_freq == 0 and self.is_master:
        self.writer.add_scalar('train/loss', batch_loss, start_iter + curr_iter)
        self.writer.add_scalar('train/pos_loss', batch_pos_loss, start_iter + curr_iter)
        self.writer.add_scalar('train/neg_loss', batch_neg_loss, start_iter + curr_iter)
        logging.info(
            "Train Epoch: {} [{}/{}], Current Loss: {:.3e}"
            .format(epoch, curr_iter,
                    len(self.data_loader) //
                    iter_size, batch_loss) +
            "\tData time: {:.4f}, Train time: {:.4f}, Iter time: {:.4f}, LR: {}".format(
                data_meter.avg, total_timer.avg - data_meter.avg, total_timer.avg, self.scheduler.get_lr()))
        data_meter.reset()
        total_timer.reset()

  def _train_iter(self, curr_iter, data_loader_iter, start_iter, iter_size, timers):
    self.model.train()
    data_meter, data_timer, total_timer = timers
    
    self.optimizer.zero_grad()
    batch_pos_loss, batch_neg_loss, batch_loss = 0, 0, 0
    data_time = 0
    total_timer.tic()
    for _iter_idx in range(iter_size):
      # Caffe iter size
      data_timer.tic()
      input_dict = data_loader_iter.next()
      data_time += data_timer.toc(average=False)

      sinput0 = ME.SparseTensor(
          input_dict['sinput0_F'], coords=input_dict['sinput0_C']).to(self.cur_device)
      F0 = self.model(sinput0).F

      sinput1 = ME.SparseTensor(
          input_dict['sinput1_F'], coords=input_dict['sinput1_C']).to(self.cur_device)

      F1 = self.model(sinput1).F

      pos_pairs = input_dict['correspondences']
      pos_loss, neg_loss = self.contrastive_hardest_negative_loss(
          F0,
          F1,
          pos_pairs,
          num_pos=self.config.trainer.num_pos_per_batch * self.batch_size,
          num_hn_samples=self.config.trainer.num_hn_samples_per_batch *
          self.batch_size)

      pos_loss /= iter_size
      neg_loss /= iter_size
      loss = pos_loss + self.neg_weight * neg_loss

      loss.backward()
      
      result = {"loss": loss, "pos_loss": pos_loss, "neg_loss": neg_loss}
      if self.config.misc.num_gpus > 1:
        result = du.scaled_all_reduce_dict(result, self.config.misc.num_gpus)
      batch_loss += result["loss"].item()
      batch_pos_loss += result["pos_loss"].item()
      batch_neg_loss += result["neg_loss"].item()

      self.optimizer.step()

      torch.cuda.empty_cache()

      total_timer.toc()
      data_meter.update(data_time)

      return batch_loss, batch_pos_loss, batch_neg_loss


class PointNCELossTrainer(ContrastiveLossTrainer):

  def __init__(
      self,
      config,
      data_loader):
    ContrastiveLossTrainer.__init__(self, config, data_loader)
    self.T = config.misc.nceT
    self.npos = config.misc.npos
    self.nneg = config.misc.nneg
    self.use_all_negatives = config.misc.use_all_negatives
    self.use_all_positives = config.misc.use_all_positives
    self.self_contrast = config.misc.self_contrast
    self.no_additional_neg = config.misc.no_additional_neg

    self.stat_freq = config.trainer.stat_freq
    self.lr_update_freq = config.trainer.lr_update_freq

  def train(self):

    assert self.config.trainer.infinite_sampler, "Only support InfSampler"

    curr_iter, epoch = 0, 0
    data_loader = self.data_loader
    data_loader_iter = self.data_loader.__iter__()
    iter_size = self.iter_size
    start_iter = 0
    data_meter, data_timer, total_timer = AverageMeter(), Timer(), Timer()
    
    total_loss = 0
    total_num = 0.0

    while (curr_iter < self.config.opt.max_iter):

      curr_iter += 1
      epoch = curr_iter / len(self.data_loader)

      if curr_iter % self.lr_update_freq == 0 or curr_iter == 1:
        lr = self.scheduler.get_lr()
        if self.is_master:
          logging.info(f" Epoch: {epoch}, LR: {lr}")
          self._save_checkpoint(epoch, 'checkpoint_'+str(curr_iter))
          self.scheduler.step()

      batch_loss = self._train_iter(curr_iter, data_loader_iter, start_iter, iter_size, [data_meter, data_timer, total_timer])

      total_loss += batch_loss
      total_num += 1

      # Print logs
      if curr_iter % self.stat_freq == 0 and self.is_master:
        self.writer.add_scalar('train/loss', batch_loss, start_iter + curr_iter)
        logging.info(
            "Train Epoch: {:.3f} [{}/{}], Current Loss: {:.3e}"
            .format(epoch, curr_iter,
                    len(self.data_loader) //
                    iter_size, batch_loss) +
            "\tData time: {:.4f}, Train time: {:.4f}, Iter time: {:.4f}, LR: {}".format(
                data_meter.avg, total_timer.avg - data_meter.avg, total_timer.avg, self.scheduler.get_lr()))
        data_meter.reset()
        total_timer.reset()


  def _train_iter(self, curr_iter, data_loader_iter, start_iter, iter_size, timers):
    data_meter, data_timer, total_timer = timers
    
    self.optimizer.zero_grad()
    batch_pos_loss, batch_neg_loss, batch_loss = 0, 0, 0
    data_time = 0
    total_timer.tic()
    for _iter_idx in range(iter_size):
      data_timer.tic()
      input_dict = data_loader_iter.next()
      data_time += data_timer.toc(average=False)

      sinput0 = ME.SparseTensor(
          input_dict['sinput0_F'], coords=input_dict['sinput0_C']).to(self.cur_device)
      F0 = self.model(sinput0).F

      sinput1 = ME.SparseTensor(
          input_dict['sinput1_F'], coords=input_dict['sinput1_C']).to(self.cur_device)
      F1 = self.model(sinput1).F

      N0, N1 = input_dict['pcd0'].shape[0], input_dict['pcd1'].shape[0]
      pos_pairs = input_dict['correspondences'].to(self.cur_device)
      
      q_unique, count = pos_pairs[:, 0].unique(return_counts=True)
      uniform = torch.distributions.Uniform(0, 1).sample([len(count)]).to(self.cur_device)
      off = torch.floor(uniform*count).long()
      cums = torch.cat([torch.tensor([0], device=self.cur_device), torch.cumsum(count, dim=0)[0:-1]], dim=0)
      k_sel = pos_pairs[:, 1][off+cums]

      q = F0[q_unique.long()]
      k = F1[k_sel.long()]
      assert q.shape[0] == k.shape[0]

      if not self.use_all_positives and self.npos < q.shape[0]:
          sampled_inds = np.random.choice(q.shape[0], self.npos, replace=False)
          q = q[sampled_inds]
          k = k[sampled_inds]
      
      npos = q.shape[0] 

      # pos logit
      logits = torch.mm(q, k.transpose(1, 0)) # npos by npos
      labels = torch.arange(npos).cuda().long()
      out = torch.div(logits, self.T)
      out = out.squeeze().contiguous()

      criterion = NCESoftmaxLoss().cuda()
      loss = criterion(out, labels)

      loss.backward(
      )  # To accumulate gradient, zero gradients only at the begining of iter_size
      result = {"loss": loss}
      if self.config.misc.num_gpus > 1:
        result = du.scaled_all_reduce_dict(result, self.config.misc.num_gpus)
      batch_loss += result["loss"].item()

    self.optimizer.step()

    torch.cuda.empty_cache()
    total_timer.toc()
    data_meter.update(data_time)
    return batch_loss
