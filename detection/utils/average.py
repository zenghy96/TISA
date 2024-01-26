class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0    # avg loss in last batch
        self.avg = 0    # avg loss in epoch
        self.sum = 0    # sum loss in epoch
        self.count = 0  # samples number

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        if self.count > 0:
            self.avg = self.sum / self.count
