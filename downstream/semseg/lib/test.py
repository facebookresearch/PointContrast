import logging
import os
import shutil
import tempfile
import warnings

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import label_binarize

from lib.utils import Timer, AverageMeter, precision_at_one, fast_hist, per_class_iu, \
    get_prediction, get_torch_device, save_predictions, visualize_results, \
    permute_pointcloud, save_rotation_pred

from MinkowskiEngine import SparseTensor


from lib.bfs.bfs import Clustering
from lib.datasets.evaluation.evaluate_semantic_instance import Evaluator as InstanceEvaluator
from lib.datasets.evaluation.evaluate_semantic_label import Evaluator as SemanticEvaluator

def print_info(iteration,
               max_iteration,
               data_time,
               iter_time,
               losses=None,
               scores=None,
               ious=None,
               hist=None,
               ap_class=None,
               class_names=None):
  debug_str = "{}/{}: ".format(iteration + 1, max_iteration)
  debug_str += "Data time: {:.4f}, Iter time: {:.4f}".format(data_time, iter_time)

  acc = hist.diagonal() / hist.sum(1) * 100
  debug_str += "\tLoss {loss.val:.3f} (AVG: {loss.avg:.3f})\t" \
      "Score {top1.val:.3f} (AVG: {top1.avg:.3f})\t" \
      "mIOU {mIOU:.3f} mAP {mAP:.3f} mAcc {mAcc:.3f}\n".format(
          loss=losses, top1=scores, mIOU=np.nanmean(ious),
          mAP=np.nanmean(ap_class), mAcc=np.nanmean(acc))
  if class_names is not None:
    debug_str += "\nClasses: " + " ".join(class_names) + '\n'
  debug_str += 'IOU: ' + ' '.join('{:.03f}'.format(i) for i in ious) + '\n'
  debug_str += 'mAP: ' + ' '.join('{:.03f}'.format(i) for i in ap_class) + '\n'
  debug_str += 'mAcc: ' + ' '.join('{:.03f}'.format(i) for i in acc) + '\n'

  logging.info(debug_str)


def average_precision(prob_np, target_np):
  num_class = prob_np.shape[1]
  label = label_binarize(target_np, classes=list(range(num_class)))
  with np.errstate(divide='ignore', invalid='ignore'):
    return average_precision_score(label, prob_np, average=None)


def test(model, data_loader, config, transform_data_fn=None):
  device = get_torch_device(config.misc.is_cuda)
  dataset = data_loader.dataset
  num_labels = dataset.NUM_LABELS
  global_timer, data_timer, iter_timer = Timer(), Timer(), Timer()
  criterion = nn.CrossEntropyLoss(ignore_index=config.data.ignore_label)
  losses, scores, ious = AverageMeter(), AverageMeter(), 0
  aps = np.zeros((0, num_labels))
  hist = np.zeros((num_labels, num_labels))

  logging.info('===> Start testing')

  global_timer.tic()
  data_iter = data_loader.__iter__()
  max_iter = len(data_loader)
  max_iter_unique = max_iter


  #------------------------------- add -------------------------------------
  VALID_CLASS_IDS = torch.FloatTensor(dataset.VALID_CLASS_IDS).long()
  eval_sem = SemanticEvaluator(dataset.CLASS_LABELS, dataset.VALID_CLASS_IDS)

  if config.misc.instance:
    evaluator = InstanceEvaluator(dataset.CLASS_LABELS_INSTANCE, dataset.VALID_CLASS_IDS_INSTANCE)
    cluster_thresh = 0.03 if config.dataset.test_point_level else 1.5
    cluster = Clustering(ignored_labels=dataset.IGNORE_LABELS_INSTANCE, thresh=cluster_thresh, closed_points=300, min_points=50)

  #-------------------------------------------------------------------------

  # Fix batch normalization running mean and std
  model.eval()

  # Clear cache (when run in val mode, cleanup training cache)
  torch.cuda.empty_cache()

  if config.test.save_prediction or config.test.test_original_pointcloud:
    if config.test.save_prediction:
      save_pred_dir = config.test.save_pred_dir
      os.makedirs(save_pred_dir, exist_ok=True)
    else:
      save_pred_dir = tempfile.mkdtemp()
    if os.listdir(save_pred_dir):
      raise ValueError(f'Directory {save_pred_dir} not empty. '
                       'Please remove the existing prediction.')

  with torch.no_grad():
    for iteration in range(max_iter):
      data_timer.tic()
      if config.data.return_transformation:
        coords, input, target, instances, transformation = data_iter.next()
      else:
        coords, input, target, instances = data_iter.next()
        transformation = None
      data_time = data_timer.toc(False)

      # Preprocess input
      iter_timer.tic()

      if config.net.wrapper_type != None:
        color = input[:, :3].int()
      if config.augmentation.normalize_color:
        input[:, :3] = input[:, :3] / 255. - 0.5
      sinput = SparseTensor(input, coords).to(device)

      # Feed forward
      inputs = (sinput,) if config.net.wrapper_type == None else (sinput, coords, color)
      soutput = model(*inputs)
      output = soutput.F

      pred = get_prediction(dataset, output, target).int()
      iter_time = iter_timer.toc(False)

      ######################################################################################
      #  Semantic Segmentation
      ######################################################################################
      #if config.dataset.test_point_level:
      #  #---------------point level--------------------
      #  vertices, _, gt_labels, gt_instances = dataset.load_input_by_index(iteration)
      #  pred_labels = pred[inverse_mapping[0].long()]
      #  eval_sem.update_confusion(VALID_CLASS_IDS[pred_labels.long()].numpy(), gt_labels)
      #  if config.misc.benchmark:
      #      eval_sem.write_to_benchmark(sceneId=scene_id, pred_ids=VALID_CLASS_IDS[pred_labels.long()].numpy())
      #else:
      #  #---------------voxel level--------------------
      #  valid_ids = (target != 255)
      #  gt_labels = target.clone().long()
      #  gt_labels[valid_ids] = VALID_CLASS_IDS[gt_labels[valid_ids]]
      #  eval_sem.update_confusion(VALID_CLASS_IDS[pred.long()].numpy(), gt_labels.numpy())
        ######################################################################################

      #if config.misc.instance:
      #  #####################################################################################
      #  #  Instance Segmentation
      #  ######################################################################################
      #  if config.dataset.test_point_level:
      #      # ---------------- point level -------------------
      #      vertices += pt_offsets.feats[inverse_mapping[0].long()].cpu().numpy()
      #      features = output.F[inverse_mapping[0].long()]
      #      instances = cluster.get_instances(vertices, features, class_mapping=VALID_CLASS_IDS)
      #      gt_ids = gt_labels * 1000 + gt_instances #invalid label_id(instance_id) 0(0), 255(any)
      #      evaluator.add_gt(gt_ids, scene_id) 
      #      evaluator.add_prediction(instances, scene_id)
      #      if config.misc.benchmark:
      #          evaluator.write_to_benchmark(scene_id=scene_id, pred_inst=instances)
      #  else:
      #      # --------------- voxel level------------------
      #      vertices = coords.cpu().numpy()[:,1:] + pt_offsets.F.cpu().numpy() / dataset.VOXEL_SIZE
      #      instances = cluster.get_instances(vertices, output.F.clone().cpu(), class_mapping=VALID_CLASS_IDS)
      #      instance_ids = instanceInfos[0]['instance_ids'] 
      #      gt_labels = target.clone()
      #      gt_labels[instance_ids == -1] = dataset.IGNORE_LABELS_INSTANCE[0] #invalid instance id is -1, map 0,1,255 labels to 0
      #      gt_labels = VALID_CLASS_IDS[gt_labels.long()]
      #      evaluator.add_gt((gt_labels*1000 + instance_ids).numpy(), scene_id) # map invalid to invalid label, which is ignored anyway
      #      evaluator.add_prediction(instances, scene_id)
      #  ######################################################################################


      if config.test.save_prediction or config.test.test_original_pointcloud:
        save_predictions(coords, pred, transformation, dataset, config, iteration, save_pred_dir)

      target_np = target.numpy()
      num_sample = target_np.shape[0]
      target = target.to(device)

      cross_ent = criterion(output, target.long())
      losses.update(float(cross_ent), num_sample)
      scores.update(precision_at_one(pred, target), num_sample)
      hist += fast_hist(pred.cpu().numpy().flatten(), target_np.flatten(), num_labels)
      ious = per_class_iu(hist) * 100

      prob = torch.nn.functional.softmax(output, dim=1)
      ap = average_precision(prob.cpu().detach().numpy(), target_np)
      aps = np.vstack((aps, ap))
      # Due to heavy bias in class, there exists class with no test label at all
      with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        ap_class = np.nanmean(aps, 0) * 100.

      if iteration % config.test.test_stat_freq == 0 and iteration > 0:
        reordered_ious = dataset.reorder_result(ious)
        reordered_ap_class = dataset.reorder_result(ap_class)
        class_names = dataset.get_classnames()
        print_info(
            iteration,
            max_iter_unique,
            data_time,
            iter_time,
            losses,
            scores,
            reordered_ious,
            hist,
            reordered_ap_class,
            class_names=class_names)

      if iteration % config.train.empty_cache_freq == 0:
        # Clear cache
        torch.cuda.empty_cache()

  global_time = global_timer.toc(False)

  reordered_ious = dataset.reorder_result(ious)
  reordered_ap_class = dataset.reorder_result(ap_class)
  class_names = dataset.get_classnames()
  print_info(
      iteration,
      max_iter_unique,
      data_time,
      iter_time,
      losses,
      scores,
      reordered_ious,
      hist,
      reordered_ap_class,
      class_names=class_names)

  if config.test.test_original_pointcloud:
    logging.info('===> Start testing on original pointcloud space.')
    dataset.test_pointcloud(save_pred_dir)

  logging.info("Finished test. Elapsed time: {:.4f}".format(global_time))

  #miou = eval_sem.evaluate_confusion()
  #miou = miou*100
  #ap = ap50 = ap25 = None
  #if config.misc.instance:
  #    ap, ap50, ap25 = evaluator.evaluate()
    

  return losses.avg, scores.avg, np.nanmean(ap_class), np.nanmean(per_class_iu(hist)) * 100
