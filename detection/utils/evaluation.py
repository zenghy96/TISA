import numpy as np
import torch
from torch.nn import functional as F
from sklearn import metrics

pts_catgory = ['L0', 'L1', 'L2', 'L3']


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
    def __init__(self, dist_thresh):
        self.meter = Meter()
        self.dist_thresh = dist_thresh
        self.dist = []

    def update(self, ann, det):
        tp, fp, fn, tn, n = 0., 0., 0., 0., 0.
        max_dist = 20
        for cat in pts_catgory:
            n += 1
            if cat in ann.keys():
                gt_pt = ann[cat]
                if cat in det:
                    det_pt = det[cat]
                    dist = np.linalg.norm(np.array(gt_pt) - np.array(det_pt))
                    self.dist.append(dist)
                    if dist <= self.dist_thresh:
                        tp += 1
                    else:
                        fp += 1
                else:
                    fn += 1
                    self.dist.append(20)
            elif cat in det:
                fp += 1
                self.dist.append(10)
            else:
                tn += 1
                self.dist.append(0)
        self.meter.update(tp, fp, tn, fn, n)
        
        if fp+fn>0:
            return False
        else:
            return True
        
    def report(self):
        meter = self.meter
        print('tp={}, fp={}, tn={}, fn={}, accuracy={:.4}, precision={:.4}, recall={:.4f}, f1={:.4f}'.format(
            meter.tp, meter.fp, meter.tn, meter.fn, meter.accuracy, meter.precision, meter.recall, meter.f1))
