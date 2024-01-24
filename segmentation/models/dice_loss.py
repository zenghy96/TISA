import numpy as np
import torch
from torch import Tensor


def dice_coeff(pred: Tensor, target: Tensor):
    # Average of Dice coefficient for all batches, or for a single mask
    smooth = 1e-6
    m1 = pred.view(-1)  # Flatten
    m2 = target.view(-1)  # Flatten
    intersection = (m1 * m2).sum()
    dice = (2. * intersection + smooth) / (m1.sum() + m2.sum() + smooth)
    return dice


def dice_loss(inp: Tensor, target: Tensor):
    return 1 - dice_coeff(inp, target)
