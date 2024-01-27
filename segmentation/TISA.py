import sys
sys.path.append('.')
import cv2
import matplotlib.pyplot as plt
import os
import numpy as np
import torch
from diffusers import DDPMScheduler
from torchvision import transforms as T
import argparse
import torch.nn.functional as F

from segmentation.models.unet import EvidentialUNet
from segmentation.models.model_tools import load_model
from segmentation.utils.uncertainty import cal_un_mask_mean, cal_preds_un
from segmentation.utils.utils import set_seed, plot_img, postprocess, plot_line
from segmentation.utils.metrics import cal_metrics
from segmentation.dataset.carotid_dataset import list_files
from segmentation.utils.pipeline_ddpm import DDPMPipeline
from segmentation.utils.measure import measure_CMIT


def parse_args():
    parser = argparse.ArgumentParser(description="Configuration for Test Model")
    # Paths
    parser.add_argument('--data_dir', type=str, default="data/carotid/target", help='Directory for data')
    parser.add_argument('--save_dir', type=str, default="outputs/carotid_demo", help='Output directory for results')
    parser.add_argument('--unet_ckpt', type=str, default="checkpoints/carotid_unet/seed_899/0599.pth.tar", help='Path to U-Net checkpoints')
    parser.add_argument('--ddpm_ckpt', type=str, default="checkpoints/carotid_ddpm", help='Path to DDPM checkpoints')

    # TISA Parameters
    parser.add_argument('--noise_steps', type=int, default=[50, 100, 150], help='')
    parser.add_argument('--run_n', type=int, default=3, help='')
    parser.add_argument('--thresh_un_model', type=float, default=0.12, help='')
    parser.add_argument('--thresh_un_pred', type=int, default=0.3, help='')

    # Model Parameters
    parser.add_argument('--n_class', type=int, default=1, help='Number of classes for classification/segmentation')
    parser.add_argument('--inp_size', type=int, default=(256, 256), help='Input size for the model')
    parser.add_argument('--inp_channel', type=int, default=1, help='Number of input channels')
    parser.add_argument('--amp', type=bool, default=True, help='Automatic Mixed Precision')
    parser.add_argument('--bilinear', type=bool, default=False, help='Bilinear upsampling')

    # System Configuration
    parser.add_argument('--device', type=int, default=0, help='Device ID for computation')
    parser.add_argument('--seed', type=int, default=899, help='Seed for random number generation')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    # 1. Initialize
    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device(f'cuda:{args.device}')
    set_seed(args.seed)

    # prepare ddpm
    pipeline = DDPMPipeline.from_pretrained(args.ddpm_ckpt)
    pipeline.to(device)
    noise_scheduler = DDPMScheduler(num_train_timesteps=1000)
    # prepare unet
    model = EvidentialUNet(n_channels=1, n_classes=1)
    model = load_model(model, args.unet_ckpt)
    model.to(device)
    model.eval()

    # paths
    img_paths = list_files(f"{args.data_dir}/demo")
    mask_paths = list_files(f"{args.data_dir}/demo_label")
    transforms = T.Compose([T.ToTensor(), T.Resize(args.inp_size, antialias=False), T.Normalize(mean=[.5], std=[.5])])
    for img_path, mask_path in zip(img_paths, mask_paths):
        name = img_path.split('/')[-1].split('.')[0]
        print('= '*6 + name + ' ='*6)
        # read original image and mask
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        h, w = img.shape
        horizontal_spacing, vertical_spacing = 0.2*w/256, 0.2*h/256
        img = transforms(img)[0]
        mask_true = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED).astype(np.uint8)
        mask_true = cv2.resize(mask_true, args.inp_size, interpolation = cv2.INTER_NEAREST) / 255.0

        uns_model = np.ones((len(args.noise_steps), args.run_n))
        uns_pred = np.ones((len(args.noise_steps),))
        pred_all = [[[] for i in range(args.run_n)] for j in range(len(args.noise_steps))]
        sample_all = [[[] for i in range(args.run_n)] for j in range(len(args.noise_steps))]
        un_map_all = [[[] for i in range(args.run_n)] for j in range(len(args.noise_steps))]

        for i, N in enumerate(args.noise_steps):
            # add noise to original image -> condition images
            noise_scheduler = DDPMScheduler(num_train_timesteps=1000)
            timesteps = torch.LongTensor([N])
            noise = torch.randn(img.shape)
            noisy_image = noise_scheduler.add_noise(img, noise, timesteps)
            conditions = [noisy_image.unsqueeze(dim=0)] * args.run_n
            conditions = torch.stack(conditions, dim=0)
            conditions = conditions.to(device)
            # # # # alignment !
            samples = pipeline(
                num_inference_steps=1000,
                # generator=torch.manual_seed(0),
                batch_size=args.run_n, 
                noisy_image=conditions, 
                start_step=N,
                output_type='numpy',
            ).images.squeeze()

            # predict
            inp = [transforms(x) for x in samples]
            inp = torch.stack(inp, dim=0)
            inp = inp.to(device=device)
            masks_pred, v, alpha, beta = model(inp)
            masks_pred = (F.sigmoid(masks_pred) > 0.5).float() 
            masks_pred = masks_pred.cpu().detach().numpy().squeeze()
            un_maps = beta / (v * (alpha - 1))
            un_maps = un_maps.cpu().detach().numpy().squeeze()        

            # plot
            plt.figure(figsize=(12, 8))
            plt.subplot(341)
            plot_img(img, title='origin')
            plt.subplot(345)
            plot_img(mask_true, title='label')
            plt.subplot(349)
            plot_img(noisy_image, title='condition')
            for j in range(args.run_n):
                # pred, un_map = postprocess(masks_pred[j], un_maps[j])
                pred, un_map = masks_pred[j], un_maps[j]
                dice, iou = cal_metrics(pred, mask_true)
                un_model = cal_un_mask_mean(un_map, pred)
                pred_all[i][j] = pred
                sample_all[i][j] = samples[j]
                un_map_all[i][j] = un_maps[j]

                plt.subplot(3, 4, j+2)
                plot_img(samples[j], title=f'smaple {j}')
                plt.subplot(3, 4, j+6)
                plot_img(pred, title=f'dice={dice:.02f},iou={iou:.02f}')
                plt.subplot(3, 4, j+10)
                plot_img(un_map, cmap='hot', title=f"un_model={un_model:.04f}")

                uns_model[i, j] = un_model
            plt.savefig(f'{args.save_dir}/TISA_{name}-{i}.png', bbox_inches='tight', dpi=400, pad_inches=0.1)
            plt.close()

            un_pred = cal_preds_un(masks_pred)
            uns_pred[i] = un_pred

        # # # selection !
        # cal avrage model uncertainty
        uns_model_avg = np.mean(uns_model, axis=1)
        print(f'uns_model:\n {uns_model.round(4)}')
        print(f'uns_model_avg: {uns_model_avg.round(4)}')
        print(f'uns_pred: {uns_pred.round(4)}')
        # find samples lower than two thresholds
        idx1 = np.where(uns_model_avg < args.thresh_un_model)[0]
        idx2 = np.where(uns_pred < args.thresh_un_pred)[0]
        idx = np.array(list(set(idx1) & set(idx2)))
        # find best
        if idx.size > 0:
            best_un_pred = uns_pred[idx].min()
            best_step = np.where(uns_pred==best_un_pred)[0][0]
            best_run = uns_model[best_step, :].argsort()[0]
            final_pred = pred_all[best_step][best_run]
            final_un_model = uns_model[best_step][best_run]
            dice, iou = cal_metrics(final_pred, mask_true)
            print(f'Best is [{best_step}, {best_run}]')
            print(f'dice={dice:.02f}, iou={iou:.02f}, un_model={final_un_model:.04f}')
        
            # measure !
            plt.figure(figsize=(12, 4))
            plt.subplot(131)
            plot_img(img, title='origin')
            plt.subplot(132)
            plot_img(mask_true, title='label')
            CMIT_label, upper_polyreg_label, lower_polyreg_label = measure_CMIT(
                cmi=mask_true.copy(), vertical_spacing=vertical_spacing, horizontal_spacing=horizontal_spacing, degree=4)
            plot_line(upper_polyreg_label, lower_polyreg_label, 20, 230)
            plt.text(50, 100,  'CMIT$_{manual}$'+f'={CMIT_label} mm', color='y', fontsize=10)
            plt.subplot(133)
            plot_img(final_pred, title='final predition and measure')
            CMIT_auto, upper_polyreg_auto, lower_polyreg_auto = measure_CMIT(
                cmi=final_pred.copy(), vertical_spacing=vertical_spacing, horizontal_spacing=horizontal_spacing, degree=4)
            plot_line(upper_polyreg_auto, lower_polyreg_auto, 20, 230)
            plt.text(30, 100,  'CMIT$_{manual}$'+f'={CMIT_auto} mm', color='y', fontsize=10)
            plt.savefig(f'{args.save_dir}/TISA_{name}_final.png', bbox_inches='tight', dpi=400, pad_inches=0.1)
            print(f'Measurement error = {CMIT_label-CMIT_auto:.3f}')
        else:
            print(f'No!')
