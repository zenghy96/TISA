import random
import numpy as np
import PIL
import PIL.ImageOps
import PIL.ImageEnhance
import PIL.ImageDraw
from PIL import Image
import albumentations as A

PARAMETER_MAX = 10


def Brightness(img, mask, min_v, max_v):
    v = random.uniform(min_v, max_v)
    return PIL.ImageEnhance.Brightness(img).enhance(v), mask
def Contrast(img, mask, min_v, max_v):
    v = random.uniform(min_v, max_v)
    return PIL.ImageEnhance.Contrast(img).enhance(v), mask
def Sharpness(img, mask, min_v, max_v):
    v = random.uniform(min_v, max_v)
    return PIL.ImageEnhance.Sharpness(img).enhance(v), mask


def TranslateX(img, mask, min_v, max_v):
    v = random.uniform(min_v, max_v)
    if random.random() < 0.5:
        v = -v
    v = int(v * img.size[0])
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, v, 0, 1, 0)), mask.transform(mask.size, PIL.Image.AFFINE, (1, 0, v, 0, 1, 0))
def TranslateY(img, mask, min_v, max_v):
    v = random.uniform(min_v, max_v)
    if random.random() < 0.5:
        v = -v
    v = int(v * img.size[1])
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, 0, 1, v)), mask.transform(mask.size, PIL.Image.AFFINE, (1, 0, 0, 0, 1, v))


def ShearX(img, mask, min_v, max_v):
    v = random.uniform(min_v, max_v)
    if random.random() < 0.5:
        v = -v
    return img.transform(img.size, PIL.Image.AFFINE, (1, v, 0, 0, 1, 0)), mask.transform(mask.size, PIL.Image.AFFINE, (1, v, 0, 0, 1, 0))
def ShearY(img, mask, min_v, max_v):
    v = random.uniform(min_v, max_v)
    if random.random() < 0.5:
        v = -v
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, v, 1, 0)), mask.transform(mask.size, PIL.Image.AFFINE, (1, 0, 0, v, 1, 0))


def Rotate(img, mask, min_v, max_v):
    v = random.randint(min_v, max_v)
    if random.random() < 0.5:
        v = -v
    return img.rotate(v), mask.rotate(v)


aug_pool = [
    (Brightness, 0.5, 0.9),
    (Contrast, 0.5, 0.9),
    (Sharpness, 0, 2),
    (TranslateX, 0.01, 0.05),
    (TranslateY, 0.01, 0.05),
    (ShearX, 0.05, 0.3),
    (ShearY, 0.05, 0.3),
    (Rotate, 10, 30)
]
# aug_pool = [
#     (TranslateX, 0.01, 0.05),
#     (TranslateY, 0.01, 0.05),
#     (ShearX, 0.05, 0.3),
#     (ShearY, 0.05, 0.3),
#     (Rotate, 10, 30)
# ]
aug_pool_shape = [TranslateX, TranslateY, ShearX, ShearY, Rotate]


class RandAugment(object):
    def __init__(self, n):
        assert n >= 1
        self.n = n

    def __call__(self, img, mask):
        ops = random.choices(aug_pool, k=self.n)
        for op, min_v, max_v in ops:
            if random.random() < 1:
                img, mask = op(img, mask, min_v=min_v, max_v=max_v)
        return img, mask
