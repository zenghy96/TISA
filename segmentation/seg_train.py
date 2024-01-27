import torch.nn.functional as F
import torch
import os
import tensorboardX
from torch import optim
import sys
import torch.nn as nn
from tqdm import tqdm
sys.path.append('.')
from torch.utils.data import DataLoader

from models.unet import EvidentialUNet
from models.model_tools import load_model
from segmentation.models.dice_loss import dice_loss
from models.evidence_loss import calculate_evidential_loss
from models.model_tools import save_model
from utils.utils import set_seed
from dataset.carotid_dataset  import CarotidDataset, list_files
from utils.metrics import cal_metrics
import argparse


def get_args():
    parser = argparse.ArgumentParser(description="Training configuration for carotid U-Net Model")
    # path
    parser.add_argument('--data_dir', type=str, default="data/carotid/source", help='Directory for data')
    parser.add_argument('--save_dir', type=str, default="checkpoints/carotid_unet", help='Path for saving checkpoints')
    parser.add_argument('--checkpoints', type=str, default="", help='Path to checkpoints')
    parser.add_argument('--resume', action='store_true', help='Flag to resume training from checkpoint')

    # model
    parser.add_argument('--n_class', type=int, default=1, help='Number of classes')
    parser.add_argument('--inp_size', type=int, default=256, help='Input size')
    parser.add_argument('--inp_channel', type=int, default=1, help='Input channel')

    # train
    parser.add_argument('--lr', type=float, default=1e-5, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-8, help='Weight decay')
    parser.add_argument('--momentum', type=float, default=0.999, help='Momentum')

    parser.add_argument('--n_epoch', type=int, default=800, help='Number of epochs')
    parser.add_argument('--save_fre', type=int, default=50, help='Save frequency')
    parser.add_argument('--val_fre', type=int, default=50, help='Validation frequency')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--train_percent', type=float, default=0.8, help='Training data percentage')
    parser.add_argument('--amp', action='store_true', help='Flag to use AMP (Automatic Mixed Precision)')
    parser.add_argument('--bilinear', action='store_true', help='Flag to use bilinear upscaling')

    parser.add_argument('--device', type=int, default=0, help='Device ID')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')

    args = parser.parse_args()
    return args


def validate(net, dataloader, device, amp):
    net.eval()
    num_val_batches = len(dataloader)
    dice_score = 0
    iou_score = 0
    # iterate over the validation set
    with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
        for image, mask_true, img_id in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
            image = image.to(device=device)
            mask_true = mask_true.to(device=device)
            mask_pred, v, alpha, beta = net(image)
            if net.n_classes == 1:
                assert mask_true.min() >= 0 and mask_true.max() <= 1, 'True mask indices should be in [0, 1]'
                mask_pred = (F.sigmoid(mask_pred) > 0.5).float() 
            else:
                mask_pred = F.one_hot(mask_pred.argmax(dim=1), net.n_classes).permute(0, 3, 1, 2).float()
                assert mask_true.min() >= 0 and mask_true.max() < net.n_classes, 'True mask indices should be in [0, n_classes['
            # compute the Dice score
            dice, iou = cal_metrics(mask_pred, mask_true)
            dice_score += dice
            iou_score += iou     
    net.train()
    dice_score = dice_score / max(num_val_batches, 1)
    iou_score = iou_score / max(num_val_batches, 1)
    return dice_score, iou_score


if __name__ == '__main__':
    args = get_args()
    
    # 1. Initialize logging
    save_dir = f'{args.save_dir}/seed_{args.seed}'
    log_dir = f"{save_dir}/log"
    os.makedirs(log_dir, exist_ok=True)
    ckpt_dir = f"{save_dir}/ckpt"
    os.makedirs(ckpt_dir, exist_ok=True)
    logger = tensorboardX.SummaryWriter(log_dir=log_dir)
    device = torch.device(f'cuda:{args.device}')
    set_seed(args.seed)

    # 2. Create dataset
    img_paths = list_files(f"{args.data_dir}/train")
    mask_paths = list_files(f"{args.data_dir}/train_label")
    n_train = int(len(img_paths)*args.train_percent)
    train_ds = CarotidDataset(img_paths[:n_train], mask_paths[:n_train], args.inp_size, args.n_classes, 'train')
    val_ds = CarotidDataset(img_paths[n_train:], mask_paths[n_train:], args.inp_size, args.n_classes, 'val')
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True)

    # 3. Set up the model, the optimizer, the loss, the learning 
    # rate scheduler and the loss scaling for AMP
    model = EvidentialUNet(
        n_channels=args.inp_channel, 
        n_classes=args.n_class, 
        bilinear=args.bilinear
    )
    optimizer = optim.RMSprop(
        model.parameters(), 
        lr=args.lr, 
        weight_decay=args.weight_decay, 
        momentum=args.momentum, 
        foreach=True
    )
    if args.checkpoints and args.resume:
        model, optimizer, start_epoch = load_model(model, args.checkpoints, optimizer)
    model.to(device=device)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5)  # goal: maximize Dice score
    grad_scaler = torch.cuda.amp.GradScaler(enabled=args.amp)
    criterion = nn.CrossEntropyLoss() if model.n_classes > 1 else nn.BCEWithLogitsLoss()

    # 4. Begin training
    for epoch in range(1, args.n_epoch + 1):
        model.train()
        epoch_loss = 0
        pbar = tqdm(train_loader, desc=f'Epoch {epoch}/{args.n_epoch}', unit='batch')
        for images, true_masks, imgs_id in pbar:
            images = images.to(device=device)
            true_masks = true_masks.to(device=device)
            with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=args.amp):
                pred, v, alpha, beta = model(images)
                loss1 = criterion(pred, true_masks)
                if model.n_classes > 1:
                    pred = F.softmax(pred, dim=1)
                else:
                    pred = F.sigmoid(pred)
                loss2 = dice_loss(pred, true_masks)
                loss3 = calculate_evidential_loss(true_masks, pred, v, alpha, beta)
                loss = loss1 + loss2 + loss3
            optimizer.zero_grad(set_to_none=True)
            grad_scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            grad_scaler.step(optimizer)
            grad_scaler.update()

            pbar.set_postfix(**{'loss': loss.item(), 'loss1=': loss1.item(), 'loss2=': loss2.item(), 'loss3=': loss3.item()})
            epoch_loss += loss.item()

        logger.add_scalar('train_loss', epoch_loss/len(train_loader), epoch)

        # Evaluation round
        if epoch % args.val_fre == 0 or epoch == args.n_epoch:
            dice, iou = validate(model, val_loader, device, args.amp)
            scheduler.step(dice)
            logger.add_scalar("dice", dice, epoch)
            logger.add_scalar("iou", iou, epoch)
        
        if epoch % args.save_fre == 0 or epoch == args.n_epoch:
            save_model(
                path=f"{ckpt_dir}/{epoch:04d}.pth.tar", 
                epoch=epoch,
                model=model,
                optimizer=optimizer,
            )
    logger.close()
        
    