import os
import time
import sys
import torch
import tensorboardX


class Logger:
    def __init__(self, args):
        """Create a summary writer logging to log_dir."""
        log_dir = args.log_dir
        self.writer = tensorboardX.SummaryWriter(log_dir=log_dir)
        self.log = open(log_dir + '/log.txt', 'w')
        self.start_line = True
        # write config
        para = dict((name, getattr(args, name)) for name in dir(args) if not name.startswith('_'))
        file_name = os.path.join(log_dir, 'cfg.txt')
        with open(file_name, 'wt') as cfg_file:
            cfg_file.write('==> torch version: {}\n'.format(torch.__version__))
            cfg_file.write('==> cudnn version: {}\n'.format(torch.backends.cudnn.version()))
            cfg_file.write('==> Cmd:\n')
            cfg_file.write(str(sys.argv))
            cfg_file.write('\n==> config:\n')
            for k in para.keys():
                if not k.startswith('_'):
                    cfg_file.write('       %s: %s\n' % (str(k), str(para[k])))

    def write(self, txt):
        if self.start_line:
            time_str = time.strftime('%d-%H-%M')
            self.log.write('{}: {}'.format(time_str, txt))
        else:
            self.log.write(txt)
            self.start_line = False
        if '\n' in txt:
            self.start_line = True
            self.log.flush()
    
    def close(self):
        self.log.close()
    
    def scalar_summary(self, tag, value, step):
        """Log a scalar variable."""
        self.writer.add_scalar(tag, value, step)
