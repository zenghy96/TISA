import matplotlib.pyplot as plt


sym = {'L0': '*', 'L1': '+', 'L2': 'o', 'L3': 'p'}


def plot_det(img=None, det=None, ann=None):
    if img is not None:
        plt.imshow(img)
    if ann is not None:
        for k, pt in ann.items():
            plt.plot(pt[0], pt[1], 'w' + sym[k], markersize=4)
    if det is not None:
        for k, pt in det.items():
            plt.plot(pt[0], pt[1], 'r' + sym[k], markersize=4)
    plt.axis('off')


def plot_det_1(axs, img=None, det=None, ann=None):
    if img is not None:
        axs.imshow(img)
    if ann is not None:
        for k, pt in ann.items():
            axs.plot(pt[0], pt[1], 'w' + sym[k], markersize=4)
    if det is not None:
        for k, pt in det.items():
            axs.plot(pt[0], pt[1], 'r' + sym[k], markersize=4)
    axs.axis('off')
