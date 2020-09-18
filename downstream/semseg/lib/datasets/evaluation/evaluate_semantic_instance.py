# Evaluates semantic instance task
# Adapted from the CityScapes evaluation: https://github.com/mcordts/cityscapesScripts/tree/master/cityscapesscripts/evaluation
# Input:
#   - path to .txt prediction files
#   - path to .txt ground truth files
#   - output file to write results to
# Each .txt prediction file look like:
#    [(pred0) rel. path to pred. mask over verts as .txt] [(pred0) label id] [(pred0) confidence]
#    [(pred1) rel. path to pred. mask over verts as .txt] [(pred1) label id] [(pred1) confidence]
#    [(pred2) rel. path to pred. mask over verts as .txt] [(pred2) label id] [(pred2) confidence]
#    ...
#
# NOTE: The prediction files must live in the root of the given prediction path.
#       Predicted mask .txt files must live in a subfolder.
#       Additionally, filenames must not contain spaces.
# The relative paths to predicted masks must contain one integer per line,
# where each line corresponds to vertices in the *_vh_clean_2.ply (in that order).
# Non-zero integers indicate part of the predicted instance.
# The label ids specify the class of the corresponding mask.
# Confidence is a float confidence score of the mask.
#
# Note that only the valid classes are used for evaluation,
# i.e., any ground truth label not in the valid label set
# is ignored in the evaluation.
#
# example usage: evaluate_semantic_instance.py --scan_path [path to scan data] --output_file [output file]

# python imports
import logging
import math
import os, sys, argparse
import inspect
from copy import deepcopy
import argparse
import numpy as np

#currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
#parentdir = os.path.dirname(currentdir)
#sys.path.insert(0,parentdir)
from lib.scannet_benchmark_utils import util_3d
from lib.scannet_benchmark_utils import util

def setup_logging():
    ch = logging.StreamHandler(sys.stdout)
    logging.getLogger().setLevel(logging.INFO)
    logging.basicConfig(
        format=os.uname()[1].split('.')[0] + ' %(asctime)s %(message)s',
        datefmt='%m/%d %H:%M:%S',
        handlers=[ch])


class Evaluator:


    # ---------- Evaluation params ---------- #
    # overlaps for evaluation
    overlaps             = np.append(np.arange(0.5,0.95,0.05), 0.25)
    # minimum region size for evaluation [verts]
    min_region_sizes     = np.array( [ 100 ] )
    # distance thresholds [m]
    distance_threshes    = np.array( [  float('inf') ] )
    # distance confidences
    distance_confs       = np.array( [ -float('inf') ] )

    def __init__(self, CLASS_LABELS, VALID_CLASS_IDS, benchmark=False):
        # ---------- Label info ---------- #
        #CLASS_LABELS = ['cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 
        #                'window', 'bookshelf', 'picture', 'counter',
        #                'desk', 'curtain', 'refrigerator', 'shower curtain',
        #                'toilet', 'sink', 'bathtub', 'otherfurniture']
        #VALID_CLASS_IDS = np.array([3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39])

        self.CLASS_LABELS = CLASS_LABELS
        self.VALID_CLASS_IDS = VALID_CLASS_IDS
        self.ID_TO_LABEL = {}
        self.LABEL_TO_ID = {}
        for i in range(len(VALID_CLASS_IDS)):
            self.LABEL_TO_ID[CLASS_LABELS[i]] = VALID_CLASS_IDS[i]
            self.ID_TO_LABEL[VALID_CLASS_IDS[i]] = CLASS_LABELS[i]

        self.pred_instances = {}
        self.gt_instances = {}
        self.benchmark = benchmark

    def evaluate_matches(self, matches):
        # results: class x overlap
        ap = np.zeros( (len(self.distance_threshes) , len(self.CLASS_LABELS) , len(self.overlaps)) , np.float )
        for di, (min_region_size, distance_thresh, distance_conf) in enumerate(zip(self.min_region_sizes, self.distance_threshes, self.distance_confs)):
            for oi, overlap_th in enumerate(self.overlaps):
                pred_visited = {}
                for m in matches:
                    for p in matches[m]['pred']:
                        for label_name in self.CLASS_LABELS:
                            for p in matches[m]['pred'][label_name]:
                                if 'filename' in p:
                                    pred_visited[p['filename']] = False
                for li, label_name in enumerate(self.CLASS_LABELS):
                    y_true = np.empty(0)
                    y_score = np.empty(0)
                    hard_false_negatives = 0
                    has_gt = False
                    has_pred = False
                    for m in matches:
                        pred_instances = matches[m]['pred'][label_name]
                        gt_instances = matches[m]['gt'][label_name]
                        # filter groups in ground truth
                        gt_instances = [ gt for gt in gt_instances if gt['instance_id']>=1000 and gt['vert_count']>=min_region_size and gt['med_dist']<=distance_thresh and gt['dist_conf']>=distance_conf ]
                        if gt_instances:
                            has_gt = True
                        if pred_instances:
                            has_pred = True

                        cur_true  = np.ones ( len(gt_instances) )
                        cur_score = np.ones ( len(gt_instances) ) * (-float("inf"))
                        cur_match = np.zeros( len(gt_instances) , dtype=np.bool )
                        # collect matches
                        for (gti,gt) in enumerate(gt_instances):
                            found_match = False
                            num_pred = len(gt['matched_pred'])
                            for pred in gt['matched_pred']:
                                # greedy assignments
                                if pred_visited[pred['filename']]:
                                    continue
                                overlap = float(pred['intersection']) / (gt['vert_count']+pred['vert_count']-pred['intersection'])
                                if overlap > overlap_th:
                                    confidence = pred['confidence']
                                    # if already have a prediction for this gt,
                                    # the prediction with the lower score is automatically a false positive
                                    if cur_match[gti]:
                                        max_score = max( cur_score[gti] , confidence )
                                        min_score = min( cur_score[gti] , confidence )
                                        cur_score[gti] = max_score
                                        # append false positive
                                        cur_true  = np.append(cur_true,0)
                                        cur_score = np.append(cur_score,min_score)
                                        cur_match = np.append(cur_match,True)
                                    # otherwise set score
                                    else:
                                        found_match = True
                                        cur_match[gti] = True
                                        cur_score[gti] = confidence
                                        pred_visited[pred['filename']] = True
                            if not found_match:
                                hard_false_negatives += 1
                        # remove non-matched ground truth instances
                        cur_true  = cur_true [ cur_match==True ]
                        cur_score = cur_score[ cur_match==True ]

                        # collect non-matched predictions as false positive
                        for pred in pred_instances:
                            found_gt = False
                            for gt in pred['matched_gt']:
                                overlap = float(gt['intersection']) / (gt['vert_count']+pred['vert_count']-gt['intersection'])
                                if overlap > overlap_th:
                                    found_gt = True
                                    break
                            if not found_gt:
                                num_ignore = pred['void_intersection']
                                for gt in pred['matched_gt']:
                                    # group?
                                    if gt['instance_id'] < 1000:
                                        num_ignore += gt['intersection']
                                    # small ground truth instances
                                    if gt['vert_count'] < min_region_size or gt['med_dist']>distance_thresh or gt['dist_conf']<distance_conf:
                                        num_ignore += gt['intersection']
                                proportion_ignore = float(num_ignore)/pred['vert_count']
                                # if not ignored append false positive
                                if proportion_ignore <= overlap_th:
                                    cur_true = np.append(cur_true,0)
                                    confidence = pred["confidence"]
                                    cur_score = np.append(cur_score,confidence)

                        # append to overall results
                        y_true  = np.append(y_true,cur_true)
                        y_score = np.append(y_score,cur_score)

                    # compute average precision
                    if has_gt and has_pred:
                        # compute precision recall curve first

                        # sorting and cumsum
                        score_arg_sort      = np.argsort(y_score)
                        y_score_sorted      = y_score[score_arg_sort]
                        y_true_sorted       = y_true[score_arg_sort]
                        y_true_sorted_cumsum = np.cumsum(y_true_sorted)

                        # unique thresholds
                        (thresholds,unique_indices) = np.unique( y_score_sorted , return_index=True )
                        num_prec_recall = len(unique_indices) + 1

                        # prepare precision recall
                        num_examples      = len(y_score_sorted)
                        try:
                            num_true_examples = y_true_sorted_cumsum[-1]
                        except:
                            num_true_examples = 0

                        precision         = np.zeros(num_prec_recall)
                        recall            = np.zeros(num_prec_recall)

                        # deal with the first point
                        y_true_sorted_cumsum = np.append( y_true_sorted_cumsum , 0 )
                        # deal with remaining
                        for idx_res,idx_scores in enumerate(unique_indices):
                            cumsum = y_true_sorted_cumsum[idx_scores-1]
                            tp = num_true_examples - cumsum
                            fp = num_examples      - idx_scores - tp
                            fn = cumsum + hard_false_negatives
                            p  = float(tp)/(tp+fp)
                            r  = float(tp)/(tp+fn)
                            precision[idx_res] = p
                            recall   [idx_res] = r

                        # first point in curve is artificial
                        precision[-1] = 1.
                        recall   [-1] = 0.

                        # compute average of precision-recall curve
                        recall_for_conv = np.copy(recall)
                        recall_for_conv = np.append(recall_for_conv[0], recall_for_conv)
                        recall_for_conv = np.append(recall_for_conv, 0.)

                        stepWidths = np.convolve(recall_for_conv,[-0.5,0,0.5],'valid')
                        # integrate is now simply a dot product
                        ap_current = np.dot(precision, stepWidths)

                    elif has_gt:
                        ap_current = 0.0
                    else:
                        ap_current = float('nan')
                    ap[di,li,oi] = ap_current
        return ap

    def compute_averages(self, aps):
        d_inf = 0
        o50   = np.where(np.isclose(self.overlaps,0.5))
        o25   = np.where(np.isclose(self.overlaps,0.25))
        oAllBut25  = np.where(np.logical_not(np.isclose(self.overlaps,0.25)))
        avg_dict = {}
        #avg_dict['all_ap']     = np.nanmean(aps[ d_inf,:,:  ])
        avg_dict['all_ap']     = np.nanmean(aps[ d_inf,:,oAllBut25])
        avg_dict['all_ap_50%'] = np.nanmean(aps[ d_inf,:,o50])
        avg_dict['all_ap_25%'] = np.nanmean(aps[ d_inf,:,o25])
        avg_dict["classes"]  = {}
        for (li,label_name) in enumerate(self.CLASS_LABELS):
            avg_dict["classes"][label_name]             = {}
            #avg_dict["classes"][label_name]["ap"]       = np.average(aps[ d_inf,li,  :])
            avg_dict["classes"][label_name]["ap"]       = np.average(aps[ d_inf,li,oAllBut25])
            avg_dict["classes"][label_name]["ap50%"]    = np.average(aps[ d_inf,li,o50])
            avg_dict["classes"][label_name]["ap25%"]    = np.average(aps[ d_inf,li,o25])
        return avg_dict

    def assign_instances_for_scan(self, scene_id):
        # get gt instances
        gt_ids = self.gt_instances[scene_id]
        gt_instances = util_3d.get_instances(gt_ids, self.VALID_CLASS_IDS, self.CLASS_LABELS, self.ID_TO_LABEL)
        # associate
        gt2pred = deepcopy(gt_instances)
        for label in gt2pred:
            for gt in gt2pred[label]:
                gt['matched_pred'] = []

        pred2gt = {}
        for label in self.CLASS_LABELS:
            pred2gt[label] = []
        num_pred_instances = 0
        # mask of void labels in the groundtruth
        bool_void = np.logical_not(np.in1d(gt_ids//1000, self.VALID_CLASS_IDS))

        # go thru all prediction masks
        for instance_id in self.pred_instances[scene_id]:
            label_id = int(self.pred_instances[scene_id][instance_id]['label_id'])
            conf = self.pred_instances[scene_id][instance_id]['conf']
            if not label_id in self.ID_TO_LABEL:
                continue
            label_name = self.ID_TO_LABEL[label_id]
            # read the mask
            pred_mask = self.pred_instances[scene_id][instance_id]['pred_mask']
            # convert to binary
            num = np.count_nonzero(pred_mask)
            if num < self.min_region_sizes[0]:
                continue  # skip if empty

            pred_instance = {}
            pred_instance['filename'] = str(scene_id) + '/' + str(instance_id)
            pred_instance['pred_id'] = num_pred_instances
            pred_instance['label_id'] = label_id
            pred_instance['vert_count'] = num
            pred_instance['confidence'] = conf
            pred_instance['void_intersection'] = np.count_nonzero(np.logical_and(bool_void, pred_mask))

            # matched gt instances
            matched_gt = []
            # go thru all gt instances with matching label
            for (gt_num, gt_inst) in enumerate(gt2pred[label_name]):
                intersection = np.count_nonzero(np.logical_and(gt_ids == gt_inst['instance_id'], pred_mask))
                if intersection > 0:
                    gt_copy = gt_inst.copy()
                    pred_copy = pred_instance.copy()
                    gt_copy['intersection']   = intersection
                    pred_copy['intersection'] = intersection
                    matched_gt.append(gt_copy)
                    gt2pred[label_name][gt_num]['matched_pred'].append(pred_copy)
            pred_instance['matched_gt'] = matched_gt
            num_pred_instances += 1
            pred2gt[label_name].append(pred_instance)

        return gt2pred, pred2gt

    def print_results(self, avgs):
        sep     = "" 
        col1    = ":"
        lineLen = 64
    
        logging.info("")
        logging.info("#"*lineLen)
        line  = ""
        line += "{:<15}".format("what"      ) + sep + col1
        line += "{:>15}".format("AP"        ) + sep
        line += "{:>15}".format("AP_50%"    ) + sep
        line += "{:>15}".format("AP_25%"    ) + sep
        logging.info(line)
        logging.info("#"*lineLen)
    
        for (li,label_name) in enumerate(self.CLASS_LABELS):
            ap_avg  = avgs["classes"][label_name]["ap"]
            ap_50o  = avgs["classes"][label_name]["ap50%"]
            ap_25o  = avgs["classes"][label_name]["ap25%"]
            line  = "{:<15}".format(label_name) + sep + col1
            line += sep + "{:>15.3f}".format(ap_avg ) + sep
            line += sep + "{:>15.3f}".format(ap_50o ) + sep
            line += sep + "{:>15.3f}".format(ap_25o ) + sep
            logging.info(line)
    
        all_ap_avg  = avgs["all_ap"]
        all_ap_50o  = avgs["all_ap_50%"]
        all_ap_25o  = avgs["all_ap_25%"]
    
        logging.info("-"*lineLen)
        line  = "{:<15}".format("average") + sep + col1 
        line += "{:>15.3f}".format(all_ap_avg)  + sep 
        line += "{:>15.3f}".format(all_ap_50o)  + sep
        line += "{:>15.3f}".format(all_ap_25o)  + sep
        logging.info(line)
        logging.info("")
    
    @staticmethod
    def write_to_benchmark(output_path='benchmark_instance', scene_id=None, pred_inst={}):
        os.makedirs(output_path, exist_ok=True)
        os.makedirs(os.path.join(output_path, 'predicted_masks'), exist_ok=True)
        f = open(os.path.join(output_path, scene_id + '.txt'), 'w')
        for instance_id in pred_inst:
            # for pred instance id starts from 0; in gt valid instance id starts from 1
            score = pred_inst[instance_id]['conf']
            label = pred_inst[instance_id]['label_id']
            mask = pred_inst[instance_id]['pred_mask']

            f.write('predicted_masks/{}_{:03d}.txt {} {:.4f}'.format(scene_id, instance_id, label, score))
            if instance_id < len(pred_inst) - 1:
                f.write('\n')
            util_3d.export_ids(os.path.join(output_path, 'predicted_masks', scene_id + '_%03d.txt' % (instance_id)), mask)
        f.close()


    def add_prediction(self, instance_info, id):
        self.pred_instances[id] = instance_info
    
    def add_gt(self, instance_info, id):
        self.gt_instances[id] = instance_info
    
    def evaluate(self):
        print('evaluating', len(self.pred_instances), 'scans...')
        matches = {}
        for i, scene_id in enumerate(self.pred_instances):
            gt2pred, pred2gt = self.assign_instances_for_scan(scene_id)
            matches[scene_id] = {}
            matches[scene_id]['gt'] = gt2pred
            matches[scene_id]['pred'] = pred2gt
            sys.stdout.write("\rscans processed: {}".format(i+1))
            sys.stdout.flush()

        print('')
        ap_scores = self.evaluate_matches(matches)
        avgs = self.compute_averages(ap_scores)

        # print
        self.print_results(avgs)
        return avgs['all_ap'], avgs['all_ap_50%'], avgs['all_ap_25%']

def write_result_file(avgs, filename):
    _SPLITTER = ','
    with open(filename, 'w') as f:
        f.write(_SPLITTER.join(['class', 'class id', 'ap', 'ap50', 'ap25']) + '\n')
        for i in range(len(VALID_CLASS_IDS)):
            class_name = CLASS_LABELS[i]
            class_id = VALID_CLASS_IDS[i]
            ap = avgs["classes"][class_name]["ap"]
            ap50 = avgs["classes"][class_name]["ap50%"]
            ap25 = avgs["classes"][class_name]["ap25%"]
            f.write(_SPLITTER.join([str(x) for x in [class_name, class_id, ap, ap50, ap25]]) + '\n')    

def config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred_path', required=True, help='path to directory of predicted .txt files')
    parser.add_argument('--gt_path', required=True, help='path to directory of gt .txt files')
    parser.add_argument('--output_file', default='semantic_instance_evaluation.txt', help='output file [default: semantic_instance_evaluation.txt]')
    opt = parser.parse_args()
    return opt

if __name__ == '__main__':
    opt = config()
    setup_logging()

    #-----------------scannet----------------------
    CLASS_LABELS = ['cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window', 
                     'bookshelf', 'picture', 'counter', 'desk', 'curtain', 'refrigerator', 
                     'shower curtain', 'toilet', 'sink', 'bathtub', 'otherfurniture']
    VALID_CLASS_IDS = np.array([3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39])
    evaluator = Evaluator(CLASS_LABELS=CLASS_LABELS, VALID_CLASS_IDS=VALID_CLASS_IDS)

    print('reading', len(os.listdir(opt.pred_path))-1, 'scans...')
    for i, pred_file in enumerate(os.listdir(opt.pred_path)):
        if os.path.isdir(os.path.join(opt.pred_path, pred_file)):
            continue
        scene_id = pred_file[:12]
        sys.stdout.write("\rscans read: {}".format(i+1))
        sys.stdout.flush()

        gt_file = os.path.join(opt.gt_path, pred_file)
        gt_ids = util_3d.load_ids(gt_file)
        evaluator.add_gt(gt_ids, scene_id)

        instances = util_3d.read_instance_prediction_file(os.path.join(opt.pred_path,pred_file), opt.pred_path)
        for pred_mask_file in instances:
            # read the mask
            pred_mask = util_3d.load_ids(pred_mask_file)
            instances[pred_mask_file]['pred_mask'] = pred_mask
            

        evaluator.add_prediction(instances, scene_id)

    print('')
    _, _, _ = evaluator.evaluate()


