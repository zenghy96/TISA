import numpy as np
from itertools import combinations


none = 1e3
pt_category = ['L0', 'L1', 'L2', 'L3']

    
def cal_image_level_uncertainty(epis, det, un_thresh):
    r"""
    Calculate the uncertainty associated with a single detection in an image.
    Why ignore:
        if L0, L1 are not detected, un_L0_L1 will be very small
        if L2, L3 are detected but un_L2_L3 are large
        the mean will be small if all uns are counted
    """
    image_level_un = 0.0
    n = 0
    for k, pt in det.items():
        x, y = pt
        index = pt_category.index(k)
        ep = epis[index]
        un = ep[y, x]
        # if un > un_thresh:
        #     uns += float('inf')
        # else:
        image_level_un += un
        n += 1
    if n>0:
        image_level_un = image_level_un/n
    return image_level_un if image_level_un < un_thresh else float('inf')


def cal_variation(dets):
    r"""
    Calculate the variation of each landmark across N detections.
    """
    num = len(dets)
    var = 0.0
    n = 0
    for cat in pt_category:
        pts = get_cat_pts_from_dets(dets, cat)
        # one category are detected on all samples: calculate var
        # one category are not detected on any sample：var is 0 and is ingored
        # one category are detected on some samples: bad generated samples, var is infinite
        if len(pts) == num:
            var += cal_pts_var(pts)
            n += 1
        elif len(pts) > 0:
            var += float('inf')
            n += 1
    if n>0:
        return var/n
    else:
        return 0


def find_best_det(args, uncertainty, variation, all_dets):
    # exclude r_step without any detection
    uncertainty[uncertainty==0] = float('inf')
    # find detections meet uncertainty and variation
    idxes1 = np.where(uncertainty < args.un_thresh)[0]
    idxes2 = np.where(variation < args.var_thresh)[0]
    idxes = np.array(list(set(idxes1) & set(idxes2)))

    # find best detection: more detected landmarks or 
    final_det, final_un, final_var, final_idx = {}, None, None, None
    for idx in idxes:
        dets = all_dets[idx]
        # for n samples of r_step, combine all detctions into one
        combined_det = {}
        for cat in pt_category:
            pts = get_cat_pts_from_dets(dets, cat)
            if len(pts) > 0:
                combined_det[cat] = find_best_pt(pts)
        # the detection which has more detected landmarks are reserved
        if len(combined_det) > len(final_det):
            final_det = combined_det
            final_un = uncertainty[idx]
            final_var = variation[idx]
            final_idx = idx
        # if 
        # elif len(combined_det) == len(final_det) and uncertainty[idx] < final_un:
        #     final_det = combined_det
        #     final_un = uncertainty[idx]
        #     final_var = variation[idx]
        #     final_idx = idx
    return final_det, final_un, final_var, final_idx


def get_cat_pts_from_dets(dets, cat):
    pts = []
    for det in dets:
        if cat in det:
            pts.append(np.array(det[cat]))
    return pts


def cal_pts_var(pts):
    num = len(pts)
    x_mean = sum([pts[i][0] for i in range(num)]) / num
    y_mean = sum([pts[i][1] for i in range(num)]) / num
    center = np.array([x_mean, y_mean])
    distance = 0.0
    for pt in pts:
        distance += np.linalg.norm(pt - center)
    return distance / num


def find_best_pt(pts):
    r"""
    Give a set of coordinates, find the best coordinate based on mean
    """
    num = len(pts)
    x_mean = sum([pts[i][0] for i in range(num)]) / num
    y_mean = sum([pts[i][1] for i in range(num)]) / num
    center = np.array([x_mean, y_mean])
    min_distance = float('inf')
    min_pt = None
    for pt in pts:
        distance = np.linalg.norm(pt - center)
        if distance < min_distance:
            min_distance = distance
            min_pt = pt.tolist()
    return min_pt


def check_final_det(det):
    r""" 
    the lamina landmarks are paired
    delete lonely predicted landmark
    """
    if 'L0' in det and 'L1' not in det:
        del det['L0']
    elif 'L0' not in det and 'L1' in det:
        del det['L1']
    if 'L2' in det and 'L3' not in det:
        del det['L2']
    elif 'L2' not in det and 'L3' in det:
        del det['L3']
    return det 
