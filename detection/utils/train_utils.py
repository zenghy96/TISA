from torch.optim.lr_scheduler import LambdaLR
import math
import random
import numpy as np
import torch
from copy import deepcopy


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, 
                                    num_cycles=7./16., last_epoch=-1):
    def _lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        no_progress = float(current_step - num_warmup_steps) / \
            float(max(1, num_training_steps - num_warmup_steps))
        return max(0., math.cos(math.pi * num_cycles * no_progress))
    return LambdaLR(optimizer, _lr_lambda, last_epoch)


def adjust_learning_rate(lr, schedule, optimizer, epoch):
    """Decay the learning rate based on schedule"""
    for milestone in schedule:
        if epoch >= milestone:
            lr *= 0.1
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return optimizer
