import os
import numpy as np
import random


def de_ann(ann):
    new = {}
    shapes = ann['shapes']
    for shape in shapes:
        label = shape['label']
        pt = shape['points'][0]
        new[label] = pt
    return new


def en_ann(dets, split, img_id, un=None, var=None):
    det_json = {
        "version": "4.5.9",
        "flags": {},
        "shapes": [],
        "imagePath": "../ALL/US47_0250.jpg",
        "imageData": None,
        "imageHeight": 480,
        "imageWidth": 640
    }
    ann = {"label": "", "points": [], "group_id": None,
           "shape_type": "point", "flags": {}}
    shapes = []
    for k, v in dets.items():
        ann["label"] = k
        ann["points"] = [v]
        shapes.append(ann.copy())
    det_json["shapes"] = shapes
    det_json["imagePath"] = "../{}/{}.jpg".format(split, img_id)

    det_json["uncertainty"] = un
    det_json["variation"] = var
    
    return det_json
