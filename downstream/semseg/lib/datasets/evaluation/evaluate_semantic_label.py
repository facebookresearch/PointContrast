# Evaluates semantic label task
# Input:
#   - path to .txt prediction files
#   - path to .txt ground truth files
#   - output file to write results to
# Note that only the valid classes are used for evaluation,
# i.e., any ground truth label not in the valid label set
# is ignored in the evaluation.
#
# example usage: evaluate_semantic_label.py --scan_path [path to scan data] --output_file [output file]

# python imports
import math
import logging
import os, sys, argparse
import inspect

try:
    import numpy as np
except:
    print("Failed to import numpy package.")
    sys.exit(-1)
try:
    from itertools import izip
except ImportError:
    izip = zip

#currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
#parentdir = os.path.dirname(currentdir)
#sys.path.insert(0,parentdir)
from lib.scannet_benchmark_utils import util_3d
from lib.scannet_benchmark_utils import util


class Evaluator:
    def __init__(self, CLASS_LABELS, VALID_CLASS_IDS):
        #CLASS_LABELS = ['wall', 'floor', 'cabinet', 'bed', 'chair', 'sofa', 'table', 
        #                'door', 'window', 'bookshelf', 'picture', 'counter', 'desk', 
        #                'curtain', 'refrigerator', 'shower curtain', 'toilet', 'sink', 'bathtub', 'otherfurniture']
        #VALID_CLASS_IDS = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39])
        self.CLASS_LABELS = CLASS_LABELS
        self.VALID_CLASS_IDS = VALID_CLASS_IDS
        self.UNKNOWN_ID = np.max(VALID_CLASS_IDS) + 1
        self.gt = {}
        self.pred = {}

        max_id = self.UNKNOWN_ID
        self.confusion = np.zeros((max_id+1, max_id+1), dtype=np.ulonglong)

    def update_confusion(self, pred_ids, gt_ids, sceneId=None):
        # sanity checks
        if not pred_ids.shape == gt_ids.shape:
            util.print_error('%s: number of predicted values does not match number of vertices' % pred_file, user_fault=True)

        n = self.confusion.shape[0]
        k = (gt_ids >= 0) & (gt_ids < n)
        temporal = np.bincount(n * gt_ids[k].astype(int) + pred_ids[k], minlength=n**2).reshape(n, n)

        for valid_class_row in self.VALID_CLASS_IDS:
            for valid_class_col in self.VALID_CLASS_IDS:
                self.confusion[valid_class_row][valid_class_col] += temporal[valid_class_row][valid_class_col]
    
    @staticmethod
    def write_to_benchmark(base='benchmark_segmentation', sceneId=None, pred_ids=None):
        os.makedirs(base, exist_ok=True)
        util_3d.export_ids('{}.txt'.format(os.path.join(base, sceneId)), pred_ids)

    def get_iou(self, label_id, confusion):
        if not label_id in self.VALID_CLASS_IDS:
            return float('nan')
        # #true positives
        tp = np.longlong(confusion[label_id, label_id])
        # #false negatives
        fn = np.longlong(confusion[label_id, :].sum()) - tp
        # #false positives
        not_ignored = [l for l in self.VALID_CLASS_IDS if not l == label_id]
        fp = np.longlong(confusion[not_ignored, label_id].sum())

        denom = (tp + fp + fn)
        if denom == 0:
            return float('nan')
        return (float(tp) / denom, tp, denom)

    def write_result_file(self, confusion, ious, filename):
        with open(filename, 'w') as f:
            f.write('iou scores\n')
            for i in range(len(self.VALID_CLASS_IDS)):
                label_id = self.VALID_CLASS_IDS[i]
                label_name = self.CLASS_LABELS[i]
                iou = ious[label_name][0]
                f.write('{0:<14s}({1:<2d}): {2:>5.3f}\n'.format(label_name, label_id, iou))
            f.write("{0:<14s}: {1:>5.3f}".format('mean', np.array([ious[k][0] for k in ious]).mean()))

            f.write('\nconfusion matrix\n')
            f.write('\t\t\t')
            for i in range(len(self.VALID_CLASS_IDS)):
                #f.write('\t{0:<14s}({1:<2d})'.format(CLASS_LABELS[i], VALID_CLASS_IDS[i]))
                f.write('{0:<8d}'.format(self.VALID_CLASS_IDS[i]))
            f.write('\n')
            for r in range(len(self.VALID_CLASS_IDS)):
                f.write('{0:<14s}({1:<2d})'.format(self.CLASS_LABELS[r], self.VALID_CLASS_IDS[r]))
                for c in range(len(self.VALID_CLASS_IDS)):
                    f.write('\t{0:>5.3f}'.format(confusion[self.VALID_CLASS_IDS[r],self.VALID_CLASS_IDS[c]]))
                f.write('\n')
        print('wrote results to', filename)

    def evaluate_confusion(self, output_file=None):
        class_ious = {}
        counter = 0
        summation = 0 

        for i in range(len(self.VALID_CLASS_IDS)):
            label_name = self.CLASS_LABELS[i]
            label_id = self.VALID_CLASS_IDS[i]
            class_ious[label_name] = self.get_iou(label_id, self.confusion)
        # print
        logging.info('classes          IoU')
        logging.info('----------------------------')
        for i in range(len(self.VALID_CLASS_IDS)):
            label_name = self.CLASS_LABELS[i]
            try:
                logging.info('{0:<14s}: {1:>5.3f}   ({2:>6d}/{3:<6d})'.format(label_name, class_ious[label_name][0], class_ious[label_name][1], class_ious[label_name][2]))
                summation += class_ious[label_name][0]
                counter += 1
            except:
                logging.info('{0:<14s}: nan     (   nan/nan   )'.format(label_name))

        logging.info("{0:<14s}: {1:>5.3f}".format('mean', summation / counter))

        if output_file:
            self.write_result_file(self.confusion, class_ious, output_file)

        return summation / counter

def config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred_path', required=True, help='path to directory of predicted .txt files')
    parser.add_argument('--gt_path', required=True, help='path to gt files')
    parser.add_argument('--output_file', type=str, default='./semantic_label_evaluation.txt')
    opt = parser.parse_args()
    return opt

def main():
    opt = config()

    #------------------------- ScanNet --------------------------
    CLASS_LABELS = ['wall', 'floor', 'cabinet', 'bed', 'chair', 'sofa', 'table', 
                    'door', 'window', 'bookshelf', 'picture', 'counter', 'desk', 
                    'curtain', 'refrigerator', 'shower curtain', 'toilet', 'sink', 'bathtub', 'otherfurniture']
    VALID_CLASS_IDS = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39])
    evaluator = Evaluator(CLASS_LABELS=CLASS_LABELS, VALID_CLASS_IDS=VALID_CLASS_IDS)

    print('reading', len(os.listdir(opt.pred_path))-1, 'scans...')
    for i, pred_file in enumerate(os.listdir(opt.pred_path)):
        if pred_file == 'semantic_label_evaluation.txt':
            continue

        gt_file = os.path.join(opt.gt_path, pred_file)
        if not os.path.isfile(gt_file):
            util.print_error('Result file {} does not match any gt file'.format(pred_file), user_fault=True)
        gt_ids = util_3d.load_ids(gt_file)

        pred_file = os.path.join(opt.pred_path, pred_file)
        pred_ids = util_3d.load_ids(pred_file)

        evaluator.update_confusion(pred_ids, gt_ids, pred_file.split('.')[0])
        sys.stdout.write("\rscans processed: {}".format(i+1))
        sys.stdout.flush()

    # evaluate
    evaluator.evaluate_confusion(opt.output_file)


if __name__ == '__main__':
    main()
