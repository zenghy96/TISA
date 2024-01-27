from matplotlib import pyplot as plt
import numpy as np
import torch.nn.functional as F
import torch
import os
import sys
sys.path.append('.')
from torch.utils.data import DataLoader
import scipy.io as sio
import argparse
from tqdm import tqdm

from models.unet import EvidentialUNet
from models.model_tools import load_model
from dataset.carotid_dataset import CarotidDataset, list_files
from utils.metrics import cal_metrics
from segmentation.utils.utils import plot_img, set_seed, postprocess
from utils.uncertainty import cal_un_mask_mean



def parse_args():
    parser = argparse.ArgumentParser(description="Configuration for Test Model")
    # Paths
    parser.add_argument('--data_dir', type=str, default="data/carotid/target", help='Directory for data')
    parser.add_argument('--output_dir', type=str, default="outputs/carotid_demo", help='Output directory for results')
    # parser.add_argument('--data_dir', type=str, default="data/carotid/source", help='Directory for data')
    # parser.add_argument('--output_dir', type=str, default="outputs/carotid_source", help='Output directory for results')
    
    parser.add_argument('--checkpoints', type=str, default="checkpoints/carotid_unet/seed_899/0599.pth.tar", help='Path to model checkpoints')

    # Model Parameters
    parser.add_argument('--n_class', type=int, default=1, help='Number of classes for classification/segmentation')
    parser.add_argument('--inp_size', type=int, default=256, help='Input size for the model')
    parser.add_argument('--inp_channel', type=int, default=1, help='Number of input channels')
    parser.add_argument('--amp', type=bool, default=True, help='Automatic Mixed Precision')
    parser.add_argument('--bilinear', type=bool, default=False, help='Bilinear upsampling')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for training')

    # System Configuration
    parser.add_argument('--device', type=int, default=0, help='Device ID for computation')
    parser.add_argument('--seed', type=int, default=899, help='Seed for random number generation')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    # 1. Initialize logging
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device(f'cuda:{args.device}')
    set_seed(args.seed)

    # 2. Create dataset
    img_paths = list_files(f"{args.data_dir}/demo")
    mask_paths = list_files(f"{args.data_dir}/demo_label")
    test_ds = CarotidDataset(img_paths, mask_paths, args.inp_size, args.n_class, 'test')
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=4, drop_last=False)
    print(f"load {len(test_ds)} test samples")

    # 3. Prepare model
    model = EvidentialUNet(n_channels=args.inp_channel, n_classes=args.n_class, bilinear=args.bilinear)
    model = load_model(model, args.checkpoints)
    model.to(device=device)
    model.eval()
    
    # 4. Iterate over the validation set
    results = []
    with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=args.amp):
        # bar = tqdm(test_loader, total=len(test_loader), desc='Test round', unit='batch')
        for index, (images, masks_true, imgs_id) in enumerate(test_loader):
            images = images.to(device=device)
            masks_true = masks_true.to(device=device)
            # predict the mask
            masks_pred, v, alpha, beta = model(images)
            un_maps = beta / (v * (alpha - 1))
            if model.n_classes == 1:
                assert masks_true.min() >= 0 and masks_true.max() <= 1, 'True mask indices should be in [0, 1]'
                masks_pred = (F.sigmoid(masks_pred) > 0.5).float() 
            else:
                masks_pred = F.one_hot(masks_pred.argmax(dim=1), model.n_classes).permute(0, 3, 1, 2).float()
                assert masks_true.min() >= 0 and masks_true.max() < model.n_classes, 'True mask indices should be in [0, n_classes['
            
            images = (images / 2 + 0.5).clamp(0, 1)
            images = images.cpu().detach().numpy().squeeze(axis=1)
            masks_true = masks_true.cpu().detach().numpy().squeeze(axis=1)
            masks_pred = masks_pred.cpu().detach().numpy().squeeze(axis=1)
            un_maps = un_maps.cpu().detach().numpy().squeeze(axis=1)
            for i in range(images.shape[0]):
                img = images[i]
                gt = masks_true[i]
                pred = masks_pred[i]
                dice, iou = cal_metrics(pred, gt)
                un_map = un_maps[i]
                un_model = cal_un_mask_mean(un_map, pred)
                results.append([dice, iou, un_model])
                print(f'{imgs_id[i]}: dice={dice:.02f}, iou={iou:.02f}, un_model={un_model:.04f}')

                # plot results
                plt.figure(figsize=(8, 8))
                plt.subplot(221)
                plot_img(img, title='origin')
                plt.subplot(222)
                plot_img(gt, title='label')
                plt.subplot(223)
                plot_img(pred, title=f'dice={dice:.02f},iou={iou:.02f}')
                plt.subplot(224)
                plot_img(un_map, cmap='hot', title=f"un_model={un_model:.04f}")
                plt.savefig(f"{output_dir}/SourceOnly_{imgs_id[i]}.png", bbox_inches='tight', dpi=400, pad_inches=0.1)
                plt.close()

                # # # # # # # # # # # # # # # #
                # save figures for paper show
                # # # # # # # # # # # # # # # #
                pic_save_dir = 'outputs/paper_show'
                os.makedirs(pic_save_dir, exist_ok=True)
                plt.figure(figsize=(4, 4))
                plt.imshow(pred, cmap='gray')
                plt.axis('off')
                plt.savefig(f'{pic_save_dir}/{imgs_id[i]}_source.png', bbox_inches='tight', dpi=400, pad_inches=0)
                plt.figure(figsize=(4, 4))
                plt.imshow(un_map, cmap='hot')
                plt.text(10, 246, f'uncertainty = {un_model:.04f}', color='y', fontsize=12)
                plt.axis('off')
                plt.savefig(f'{pic_save_dir}/{imgs_id[i]}_source-un.png', bbox_inches='tight', dpi=400, pad_inches=0)

    results = np.array(results)
    print('= '*7 + 'average results' + ' ='*7)
    print(f"dice = {results[:, 0].mean()}")  
    print(f"iou = {results[:, 1].mean()}")  
    print(f"un = {results[:, 2].mean()}")  
    