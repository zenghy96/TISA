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
import matplotlib.pyplot as plt
import cv2


parser = argparse.ArgumentParser(description='pts')
# path
parser.add_argument(
    '--data_dir', 
    type=str, 
    default='datasets/ClariusDetData/data'
)
parser.add_argument(
    '--sample_dir', 
    type=str, 
    default='datasets/ClariusDetData/sample'
)
parser.add_argument(
    '--ann_dir',
    type=str, 
    default='datasets/ClariusDetData/annotation'
)
parser.add_argument(
    '--output_dir', 
    type=str, 
    default='outputs/detection'
)
parser.add_argument(
    '--ckpt_path', 
    type=str, 
    default='ckpt/det/model_0500.pth.tar'
)
parser.add_argument(
    '--hm_thresh', 
    type=float, 
    default=0.5,
)
parser.add_argument(
    '--r_step_num', 
    type=int, 
    default=4,
)
parser.add_argument('--sigma', type=int, default=2)
parser.add_argument('--stacks', type=int, default=2)
parser.add_argument('--hg_order', type=int, default=4)
parser.add_argument('--inp_size', type=int, default=(256, 256))
parser.add_argument('--oup_size', type=int, default=(64, 64))
parser.add_argument('--inp_dim', type=int, default=256)
parser.add_argument('--oup_dim', type=int, default=4)
parser.add_argument('--seed', type=int, default=3)
parser.add_argument('--workers', type=int, default=2)
parser.add_argument('--devices', type=str, default='2')


def main():
    # initialize
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.devices
    devices = args.devices.split(',')
    args.device_ids = [i for i in range(len(devices))]
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    r_step_num = args.r_step_num
    
    # data
    sample_data = SampleData(args.data_dir, args.sample_dir, args.ann_dir)

    # model
    model = StackedHourglass(args.stacks, args.hg_order, args.inp_dim, args.oup_dim)
    model = load_model(model, args.ckpt_path)
    model.cuda()
    model = torch.nn.DataParallel(model, device_ids=args.device_ids)
    model.eval()

    # Start predict
    progress_bar = tqdm(total=len(sample_data))
    for image, samples, ann, img_id in sample_data:
        # model output
        inp = samples.cuda()
        with torch.no_grad():
            pred, v, alpha, beta = model(inp)
            hms = pred[-1].cpu().detach().numpy()
            v = v[-1].cpu().detach().numpy()
            alpha = alpha[-1].cpu().detach().numpy()
            beta = beta[-1].cpu().detach().numpy()
            epis = beta / (v * (alpha - 1))
        samples = np.array_split(samples.numpy().squeeze(), r_step_num, axis=0)
        hms = np.array_split(hms, r_step_num, axis=0)
        epis = np.array_split(epis, r_step_num, axis=0)

        # deocode
        h, w = image.shape
        _ann = pts_affine(ann, 256/h, 256/w)
        image = cv2.resize(image, (256, 256))
        for r, (sample, hm, ep) in enumerate(zip(samples, hms, epis)):
            plt.figure(figsize=(9, 9))
            plt.subplot(2, 3, 1)
            plot_det(image, ann=_ann)
            for i in range(sample.shape[0]):
                det = decode_heatmaps(hm[i], args.hm_thresh)
                det = pts_affine(det, 256/64, 256/64)
                plt.subplot(2, 3, i+2)
                plot_det(sample[i], det, _ann)
            plt.savefig(f"{args.output_dir}/{img_id}_{r}.png")
        progress_bar.update(1)


class SampleData(Dataset):
    def __init__(self, data_dir, sample_dir, ann_dir):
        self.image_paths = _list_image_files_recursively(data_dir)
        self.sample_paths = _list_image_files_recursively(sample_dir)
        self.ann_paths = _list_image_files_recursively(ann_dir)
        self.sample_transform = self.transforms = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[.5], std=[.5]),
        ])
    
    def __len__(self):
        return len(self.ann_paths)
            
    def __getitem__(self, index):
        image_path = self.image_paths[index]
        sample_path = self.sample_paths[index]
        ann_path = self.ann_paths[index]
        img_id = sample_path.split('/')[-1].split('.')[0]
        image = sio.loadmat(image_path)['img'].astype(np.uint8)
        samples = sio.loadmat(sample_path)['sample']
        samples = [self.sample_transform(sample) for sample in samples]
        samples = torch.stack(samples, dim=0)
        with open(ann_path, 'r') as f:
            ann = json.load(f)
            ann = de_ann(ann)
        return image, samples, ann, img_id
 

if __name__ == '__main__':
    main()
