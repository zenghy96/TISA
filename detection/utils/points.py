import numpy as np
import random
import cv2
import matplotlib.pyplot as plt


class HorizontalFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, pts, class_labels):
        reverse = {'SP': 'SP', 'L0': 'L3', 'L1': 'L2', 'L2': 'L1', 'L3': 'L0'}
        if random.random() < self.p:
            img = cv2.flip(img, flipCode=1)
            _, w = img.shape
            for idx in range(len(pts)):
                pts[idx][0] = w - pts[idx][0]
                label = class_labels[idx]
                class_labels[idx] = reverse[label]
        return img, pts, class_labels


def pts_affine(pts, kh, kw):
    trans = np.array([[kw, 0], [0, kh]])
    for k, v in pts.items():
        if isinstance(v[0], list):
            x_ori, y_ori = v[0]
            v_affine = np.dot(trans, np.array([[x_ori, y_ori]]).T)
            pts[k] = [[v_affine[0, 0], v_affine[1, 0]], v[1]]
        else:
            x_ori, y_ori = v
            v_affine = np.dot(trans, np.array([[x_ori, y_ori]]).T)
            pts[k] = [v_affine[0, 0], v_affine[1, 0]]
    return pts


def replace_pts(pts):
    idx = np.lexsort((pts[1, :], pts[0, :]))
    return pts[:, idx]


def plot_pts(pts, img, color='r'):
    sym = ['x', '*', '+', 'o', 'p']
    plt.figure()
    plt.imshow(img)
    for i in range(5):
        if pts[2, i]:
            plt.plot(pts[0, i], pts[1, i], color + sym[i])
    plt.show()
