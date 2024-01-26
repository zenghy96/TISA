import sys
sys.path.append(".")
import warnings
warnings.filterwarnings("ignore")
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import os
from utils.logger import Logger
from models.model_utils import save_model, load_model
from tqdm import tqdm
import argparse
from utils.train_utils import set_seed
from utils.train_utils import adjust_learning_rate
from detection.dataset.spine_dataset import SpineDataset
from utils.average import AverageMeter
from models.SHN_evidence import StackedHourglass


def get_args():
    parser = argparse.ArgumentParser(description='spine structures detection')
    # path
    parser.add_argument('--data_dir', type=str, default='data/spine/source')
    parser.add_argument('--log_dir', type=str, default='checkpoints/')
    parser.add_argument('--checkpoint', type=str, default=None)

    # train
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--max_epoch', type=int, default=500)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--schedule', type=int, default=[60, 150], nargs='+')
    parser.add_argument('--save_fre', type=int, default=50)

    parser.add_argument('--sigma', type=int, default=2, help='Gaussian std on heatmap')
    parser.add_argument('--stacks', type=int, default=2)
    parser.add_argument('--hg_order', type=int, default=4)
    parser.add_argument('--inp_size', type=int, default=(256, 256))
    parser.add_argument('--oup_size', type=int, default=(64, 64))
    parser.add_argument('--inp_dim', type=int, default=256)
    parser.add_argument('--oup_dim', type=int, default=4)

    # device
    parser.add_argument('--seed', type=int, default=3)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--device', type=int, default=0)
    
    args = parser.parse_args()
    return args


def main():
    # 1. Initialize
    args = get_args()
    device = torch.device(f'cuda:{args.device}')
    logger = Logger(args)
    set_seed(args.seed)
    print('start training')

    # 2. Prepare
    train_ds = SpineDataset(args.data_dir, 'train', args.sigma, args.inp_size, args.oup_size,)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True, drop_last=True)
    # model, optimizer
    model = StackedHourglass(args.stacks, args.hg_order, args.inp_dim, args.oup_dim)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    start_epoch = 0
    if args.checkpoint is not None:
        model, optimizer, start_epoch = load_model(
            model=model,
            model_path=args.checkpoint,
            optimizer=optimizer,
            resume=True,
            start_lr=args.lr
        )
    model.to(device)

    # start training
    for epoch in range(start_epoch+1, args.max_epoch+1):
        optimizer = adjust_learning_rate(args.lr, args.schedule, optimizer, epoch)
        losses = AverageMeter()
        lr = optimizer.param_groups[0]['lr']
        bar = tqdm(train_loader, desc='train [{:03}/{:03}]'.format(epoch, args.max_epoch), ncols=0)

        for index, (image, hms) in enumerate(bar):
            inp, hms = image.cuda(), hms.cuda()
            _, _, _, _, loss = model(inp, hms)
            loss = loss.mean()
            losses.update(loss.item())
            del inp, image, hms
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            bar.set_postfix_str('lr={:.0e}, loss={:.6f}'.format(lr, losses.avg))

        logger.scalar_summary('loss_train', losses.avg, epoch)
        logger.write('[{:03}]: lr={:.0e} | '.format(epoch, lr))
        logger.write('loss_train {:6f} | \n'.format(losses.avg))

        if epoch % args.save_fre == 0:
            save_model(
                path=os.path.join(args.log_dir, 'model_{:04}.pth.tar'.format(epoch)),
                epoch=epoch,
                model=model,
                optimizer=optimizer
            )
    logger.close()


if __name__ == '__main__':
    main()
