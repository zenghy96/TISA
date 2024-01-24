import numpy as np


def cal_un_mask_mean(un_map, mask):
    rows, cols = un_map.shape
    total_sum = 0
    count = 0
    un_map = (un_map - un_map.min()) / (un_map.max() - un_map.min())
    for i in range(rows):
        for j in range(cols):
            if mask[i, j] == 1:
                total_sum += un_map[i, j]
                count += 1
    if count > 0:
        mean = total_sum / count
        return mean
    else:
        return 0
    

def cal_preds_un(masks_pred):
    run_n = masks_pred.shape[0]
    predictions = masks_pred.reshape(run_n, -1)
    intersection = np.prod(predictions, axis=0).sum()
    union = (np.sum(predictions, axis=0) > 0).sum()
    un_pred = 1 - intersection / union
    return un_pred