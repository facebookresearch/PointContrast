import open3d as o3d  # prevent loading error

import sys
import os
import json
import logging
import torch
from omegaconf import OmegaConf

from easydict import EasyDict as edict

from lib.ddp_data_loaders import make_data_loader
import lib.multiprocessing as mpu
import hydra

from lib.ddp_trainer import HardestContrastiveLossTrainer, PointNCELossTrainer

ch = logging.StreamHandler(sys.stdout)
logging.getLogger().setLevel(logging.INFO)
logging.basicConfig(
    format='%(asctime)s %(message)s', datefmt='%m/%d %H:%M:%S', handlers=[ch])

torch.manual_seed(0)
torch.cuda.manual_seed(0)

logging.basicConfig(level=logging.INFO, format="")

def get_trainer(trainer):
  if trainer == 'HardestContrastiveLossTrainer':
    return HardestContrastiveLossTrainer
  elif trainer == 'PointNCELossTrainer':
    return PointNCELossTrainer
  else:
    raise ValueError(f'Trainer {trainer} not found')

@hydra.main(config_path='config', config_name='defaults.yaml')
def main(config):
  logger = logging.getLogger()
  if config.misc.config:
    resume_config = OmegaConf.load(config.misc.config)
    if config.misc.weight:
      weight = config.misc.weight
      config = resume_config
      config.misc.weight = weight
    else:
      config = resume_config

  logging.info('===> Configurations')
  logging.info(config.pretty())

  # Convert to dict
  import ipdb; ipdb.set_trace()
  if config.misc.num_gpus > 1:
      mpu.multi_proc_run(config.misc.num_gpus,
              fun=single_proc_run, fun_args=(config,))
  else:
      single_proc_run(config)

def single_proc_run(config):
  train_loader = make_data_loader(
      config,
      config.trainer.train_phase,
      config.trainer.batch_size,
      num_threads=config.misc.train_num_thread,
      inf_sample=config.trainer.infinite_sampler)

  Trainer = get_trainer(config.trainer.trainer)
  trainer = Trainer(
      config=config,
      data_loader=train_loader,
  )
  trainer.train()


if __name__ == "__main__":
  main()
