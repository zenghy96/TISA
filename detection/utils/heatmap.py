import numpy as np
import torch

pts_category = ['L0', 'L1', 'L2', 'L3']


def draw_heatmaps(oup_size, sigma, pts, class_labels):
    h, w = oup_size
    heatmaps = torch.zeros((len(pts_category), h, w))
    for pt, label in zip(pts, class_labels):
        idx = pts_category.index(label)
        y_range = [i for i in range(h)]
        x_range = [i for i in range(w)]
        xx, yy = np.meshgrid(x_range, y_range)
        d = (xx - pt[0]) ** 2 + (yy - pt[1]) ** 2
        exponent = d / 2.0 / sigma / sigma
        heatmaps[idx] = torch.Tensor(np.exp(-exponent))
    return heatmaps


def decode_heatmaps(hms, vis_t):
    dets = {}
    for i, cat in enumerate(pts_category):
        hm = hms[i]
        score = hm.max()
        if score > vis_t:
            m, n = hm.shape
            index = int(hm.argmax())
            y = index // n
            x = index % n
            dets[cat] = [x, y]
    return dets
