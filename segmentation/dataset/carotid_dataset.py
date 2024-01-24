import matplotlib.pyplot as plt
import numpy as np
import cv2
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms as T
import sys
sys.path.append('.')
from dataset.randaugment import RandAugment
import torch.nn.functional as F
import blobfile as bf
import torch


class CarotidDataset(Dataset):
    def __init__(self, img_paths, mask_paths, resolution, n_classes, split='train'):
        super(CarotidDataset).__init__()
        self.img_paths = img_paths
        self.mask_paths = mask_paths
        self.split = split
        self.n_classes = n_classes
        self.resolution = (resolution, resolution)
        self.augment = RandAugment(n=2)
        self.transforms = T.Compose([
            T.ToTensor(),
            T.Resize((resolution, resolution), antialias=False),
            T.Normalize(mean=[.5], std=[.5]),
        ])
        self.mask_transforms = T.Compose([
            T.ToTensor(),
        ])
    
    def __len__(self):
        return len(self.img_paths)
        
    def __getitem__(self, index):
        img_path = self.img_paths[index]
        mask_path = self.mask_paths[index]
        assert img_path.split('/')[-1] == mask_path.split('/')[-1]
        img = Image.open(img_path)
        mask = Image.open(mask_path)
        img_id = img_path.split('/')[-1].split('.')[0]
        if self.split == 'train':
            img, mask = self.augment(img, mask)
        img = np.asarray(img, np.uint8)
        img = self.transforms(img.copy())
        mask = np.asarray(mask, np.uint8)
        mask = cv2.resize(mask, self.resolution, interpolation=cv2.INTER_NEAREST) / 255.0
        mask = torch.from_numpy(mask).unsqueeze(0)
        return img, mask, img_id


def list_files(data_dir):
    results = []
    for entry in sorted(bf.listdir(data_dir)):
        full_path = bf.join(data_dir, entry)
        results.append(full_path)
    return results


if __name__ == "__main__":
    img_paths = list_files("data/CUBS/train")
    mask_paths = list_files("data/CUBS/train_label")
    dataset = CarotidDataset(img_paths, mask_paths, 256, 'test')
    img, mask, img_id = dataset[50]
    plt.figure()
    plt.imshow(img[0])
    plt.imshow(mask[0], alpha=0.2)
    plt.savefig('cubs.png')
    plt.close()