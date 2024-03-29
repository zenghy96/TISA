import random
from matplotlib import legend, pyplot as plt
import numpy as np
import torch
import cv2

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def plot_img(img, cmap='gray', title=None):
    plt.imshow(img, cmap=cmap)
    plt.axis('off')
    if title is not None:
        plt.title(title)


def postprocess(image, un_map):
    _, binary = cv2.threshold(image, 0, 1, cv2.THRESH_BINARY)
    binary = binary.astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8, cv2.CV_32S)
    if num_labels > 2:
        volume = stats[:, -1]
        idx = volume.argsort()[::-1]
        for delete_label in idx[2:]:
            image[labels==delete_label] = 0
            un_map[labels==delete_label] = 0
    return image, un_map


def plot_line(upper_polyreg, lower_polyreg, show_start, show_end):
    x_plot = np.linspace(show_start, show_end, 500)
    y_plot_upper = upper_polyreg.predict(x_plot.reshape(-1, 1))
    y_plot_lower = lower_polyreg.predict(x_plot.reshape(-1, 1))
    plt.plot(x_plot, y_plot_upper, 'r--', linewidth=2, label='LI')
    plt.plot(x_plot, y_plot_lower, 'g--', linewidth=2, label='MA')
