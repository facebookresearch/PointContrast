import open3d as o3d  # prevent loading error

import sys
import json
import logging
import torch
from easydict import EasyDict as edict

from lib.ddp_data_loaders import make_data_loader
from config import get_config
import lib.multiprocessing as mpu
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

def main(config, resume=False):
    if config.num_gpus > 1:
        mpu.multi_proc_run(config.num_gpus,
                fun=single_proc_run, fun_args=(config,resume))
    else:
        single_proc_run(config, resume)

def single_proc_run(config, resume=False):
  train_loader = make_data_loader(
      config,
      config.train_phase,
      config.batch_size,
      num_threads=config.train_num_thread,
      inf_sample=True)

  Trainer = get_trainer(config.trainer)
  trainer = Trainer(
      config=config,
      data_loader=train_loader,
  )
  trainer.train()


if __name__ == "__main__":
  logger = logging.getLogger()
  config = get_config()

  dconfig = vars(config)
  if config.resume_dir:
    resume_config = json.load(open(config.resume_dir + '/config.json', 'r'))
    for k in dconfig:
      if k not in ['resume_dir'] and k in resume_config:
        dconfig[k] = resume_config[k]
    dconfig['resume'] = resume_config['out_dir'] + '/checkpoint.pth'

  logging.info('===> Configurations')
  for k in dconfig:
    logging.info('    {}: {}'.format(k, dconfig[k]))

  # Convert to dict
  config = edict(dconfig)
  main(config)
