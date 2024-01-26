import numpy as np
import torch
from torch.nn import functional as F
from sklearn import metrics

pts_label = ['L0', 'L1', 'L2', 'L3']


class CalPCK:
    def __init__(self):
        self.count = 0.
        self.correct_num = 0.
        self.PCK = 0.

    def update(self, gts, dets, thresh):
        error = {}
        for label, gt_pt in gts.items():
            self.count += 1
            if label in dets.keys():
                det_pt = dets[label]
                dist = np.linalg.norm(np.array(gt_pt) - np.array(det_pt))
                if dist < thresh:
                    self.correct_num += 1
                else:
                    error[label] = dist
        if self.count > 0:
            self.PCK = self.correct_num / self.count
        return error

    def report(self):
        print('PCK = {}'.format(self.PCK))


class Meter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.tp, self.fn, self.fp, self.tn = 0, 0, 0, 0
        self.count = 0.0
        self.accuracy = 0.0
        self.precision = 0.0
        self.recall = 0.0
        self.f1 = 0.0

    def update(self, tp, fp, tn, fn, n):
        self.tp += tp
        self.fn += fn
        self.fp += fp
        self.tn += tn
        self.count += n
        if (tp + fp) > 0:
            self.precision = self.tp / (self.tp + self.fp)
        if (tp + fn) > 0:
            self.recall = self.tp / (self.tp + self.fn)
        if (self.precision+self.recall) > 0:
            self.f1 = 2*self.precision*self.recall / (self.precision+self.recall)
        assert (self.tp+self.fp+self.tn+self.fn) == self.count
        if self.count > 0:
            self.accuracy = (self.tp+self.tn) / self.count


class PtsEval:
    def __init__(self):
        self.meter = Meter()

    def update(self, gts, dets, thresh):
        ret = []
        for label in pts_label:
            tp, fp, fn, tn = 0., 0., 0., 0.
            if label in gts.keys():
                gt_pt = gts[label]
                if dets[label] is not None:
                    det_pt, det_score = dets[label]
                    dist = np.linalg.norm(np.array(gt_pt) - np.array(det_pt))
                    if dist <= thresh:
                        tp += 1
                    else:
                        fp += 1
                else:
                    fn += 1
            elif dets[label] is not None:
                fp += 1
            else:
                tn += 1
            self.meter.update(tp, fp, tn, fn, 1)
        return ret

    def report(self):
        meter = self.meter
        print('tp={}, fp={}, tn={}, fn={}, accuracy={:.4}, precision={:.4}, recall={:.4f}, f1={:.4f}'.format(
            meter.tp, meter.fp, meter.tn, meter.fn, meter.accuracy, meter.precision, meter.recall, meter.f1))
