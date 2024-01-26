import sys
sys.path.append(".")
import warnings
warnings.filterwarnings("ignore")
import torch
import os
from models.model_utils import load_model
from models.SHN_evidence import StackedHourglass
import argparse
from utils.train_utils import set_seed
from utils.data_utils import de_ann
import json
import scipy.io as sio
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset
from detection.dataset.spine_dataset import _list_image_files_recursively
from torchvision import transforms as T
from utils.heatmap import decode_heatmaps
from utils.points import pts_affine
from utils.show import plot_det
from utils.train_utils import set_seed
import matplotlib.pyplot as plt
import cv2
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from detection.utils.evaluation import PtsEval
from detection.utils.data_utils import en_ann
from detection.dataset.spine_dataset import SpineDataset
from detection.utils.uncertainty import cal_image_level_uncertainty


def get_args():
    parser = argparse.ArgumentParser(description='spine structures detection')
    # path
    parser.add_argument('--data_dir', type=str, default='data/spine/source')
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--dist_thres', type=float, default=10)


    # mdoel 
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


data_dir='datasets/ClariusDetData'
det_save_dir='final_outputs/test_final_paper_show/source_det_seed899'
ckpt_path='ckpt/det_seed_899/model_0500.pth.tar'
hm_thresh=0.5


if __name__ == '__main__':
    # 1. Initialize
    args = get_args()
    set_seed(args.seed)
    os.makedirs(det_save_dir, exist_ok=True)
    device = torch.device(f'cuda:{args.device}')

    # 2. Prepare
    test_ds = SpineDataset(data_dir=data_dir, split='test', sigma=2, inp_size=256, oup_size=64)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=8, shuffle=False, num_workers=4, pin_memory=True, drop_last=False,)
    model = StackedHourglass(args.stacks, args.hg_order, args.inp_dim, args.oup_dim)
    model = load_model(model, ckpt_path)
    model.to(device)
    model.eval()

    # Start predict
    evaluator = PtsEval(dist_thresh=10)
    with torch.no_grad():
        for imgs, idxes in tqdm(test_loader):
            inputs = imgs.to(device)
            pred, v, alpha, beta = model(inputs)
            hms = pred[-1].cpu().detach().numpy()
            v = v[-1].cpu().detach().numpy()
            alpha = alpha[-1].cpu().detach().numpy()
            beta = beta[-1].cpu().detach().numpy()
            un_maps = beta / (v * (alpha - 1))

            for i in range(imgs.shape[0]):
                idx = idxes[i]
                img_id = test_ds.imgs_id[idx]
                img_size = test_ds.imgs_size[idx]
                gt = test_ds.anns[idx]
                det = decode_heatmaps(hms[i], vis_t=0.5)
                un = cal_image_level_uncertainty(un_maps[i], det, 100)
                det = pts_affine(det, img_size[0]/64, img_size[1]/64)
                evaluator.update(ann=gt, det=det)

                # save detection
                det_json = en_ann(det, 'test', img_id)
                with open(f"{det_save_dir}/{img_id}.json", 'w') as f:
                    json.dump(det_json, f, indent=4)

                # un
                # plt.figure()
                # plot_det(target_dataset.images[idx], ann=gt, det=det)
                # plt.title(un)
                # plt.savefig(f"{output_dir}/{img_id}.png")
    evaluator.report()


# class UltraDataset(Dataset):
#     def __init__(self, data_dir, split, sigma, inp_size, oup_size):
#         self.images = []
#         self.anns = []
#         self.imgs_id = []
#         self.imgs_size = []
#         img_dir = os.path.join(data_dir, 'data')
#         ann_dir = os.path.join(data_dir, 'annotation_final')

#         for filename in os.listdir(ann_dir):
#             with open(os.path.join(ann_dir, filename), 'r') as f:
#                 label = json.load(f)['shapes']
#                 ann = {}
#                 for i in range(len(label)):
#                     cat = label[i]['label']
#                     ann[cat] = label[i]['points'][0]
#                 self.anns.append(ann)
#             img_path = os.path.join(img_dir, filename.replace('.json', '.mat'))
#             img = sio.loadmat(img_path)['img'].astype(np.uint8)
#             self.images.append(img)
#             self.imgs_size.append(img.shape)
#             self.imgs_id.append(filename.split('.')[0])
#             # if len(self.images) == 400:
#             #     break
#         self.split = split
#         print('loaded {} {} samples '.format(split, len(self.anns)))
#         self.transforms = T.Compose([
#             T.ToTensor(), T.Resize((inp_size, inp_size), antialias=True),
#             T.Normalize(mean=[.5], std=[.5])])

#     def __len__(self):
#         return len(self.anns)

#     def __getitem__(self, item):
#         img = self.images[item]
#         ann = self.anns[item]
#         inp = self.transforms(img)
#         return inp, item
 

