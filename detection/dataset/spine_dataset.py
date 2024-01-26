import os
import scipy.io as sio
from torch.utils import data
import albumentations as A
from torchvision import transforms as T
import matplotlib.pyplot as plt
import json
from detection.utils.heatmap import draw_heatmaps
from detection.utils.points import HorizontalFlip
from detection.utils.data_utils import de_ann
import numpy as np


class SpineDataset(data.Dataset):
    def __init__(self, data_dir, split, sigma, inp_size, oup_size):
        self.images = []
        self.anns = []
        self.imgs_id = []
        self.imgs_size = []
        self.oup_size = oup_size
        img_dir = os.path.join(data_dir, split)
        ann_dir = os.path.join(data_dir, split+'_label')

        for filename in os.listdir(ann_dir):
            with open(os.path.join(ann_dir, filename), 'r') as f:
                label = json.load(f)['shapes']
                ann = {}
                for i in range(len(label)):
                    cat = label[i]['label']
                    ann[cat] = label[i]['points'][0]
                self.anns.append(ann)
            img_path = os.path.join(img_dir, filename.replace('.json', '.mat'))
            img = sio.loadmat(img_path)['img'].astype(np.uint8)
            self.images.append(img)
            self.imgs_size.append(img.shape)
            self.imgs_id.append(filename.split('.')[0])
        self.sigma = sigma
        self.split = split
        print('loaded {} {} samples '.format(split, len(self.anns)))

        if split == 'train':
            self.HorizontalFlip = HorizontalFlip(p=0.5)
            self.aug = A.Compose([
                A.Posterize(num_bits=7, p=0.4),
                A.Sharpen(p=0.3),
                A.GaussianBlur(blur_limit=(3, 5), p=0.2),
                A.Rotate(limit=15, p=0.5),
                A.Affine(translate_percent={"x": (-0.1, 0.1), "y": (0, 0)}, p=0.5),
                A.Affine(translate_percent={"x": (0, 0), "y": (-0.1, 0.1)}, p=0.5),
            ], keypoint_params=A.KeypointParams(format='xy'))
        self.pts_affine = A.Compose(
            [A.Resize(height=oup_size[0], width=oup_size[1])],
            keypoint_params=A.KeypointParams(format='xy')
        )
        self.transforms = T.Compose([
            T.ToTensor(), T.Resize((inp_size, inp_size)),
            T.Normalize(mean=[.5], std=[.5])])

    def __len__(self):
        return len(self.anns)

    def __getitem__(self, item):
        img = self.images[item]
        ann = self.anns[item]
        img_id = self.imgs_id[item]
        pts = []
        class_labels = []
        for i in range(len(ann)):
            pts.append(ann[i]['points'][0])

        if self.split == 'train':
            # image, pts should change with image aug
            img, pts, class_labels = self.HorizontalFlip(img, pts, class_labels)
            trans = self.aug(image=img, keypoints=pts, class_labels=class_labels)
            img, pts, class_labels = trans['image'], trans['keypoints'], trans['class_labels']
            inp = self.transforms(img)
            
            # preprocess pts and draw heatmap
            trans = self.pts_affine(image=img, keypoints=pts, class_labels=class_labels)
            pts, class_labels = trans['keypoints'], trans['class_labels']
            hms = draw_heatmaps(self.oup_size, self.sigma, pts, class_labels)
            return inp, hms
        else:
            inp = self.transforms(img)
            return inp, item

