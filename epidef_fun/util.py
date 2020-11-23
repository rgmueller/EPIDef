import numpy as np
import random
import os
from PIL import Image


def get_list_ids(lf_directory):
    """

    :param lf_directory:
    :return: list_ids:
    """
    good = 0
    scratch = 0
    dent = 0
    error = 0
    list_ids = []
    for path, subdirs, files in os.walk(lf_directory):
        if '0002_Set0_Cam_003_img.png' in files:  # only add paths with images
            list_ids.append(path)
            if 'good' in path:
                good += 1
            elif 'scratch' in path:
                scratch += 1
            elif 'dent' in path:
                dent += 1
            else:
                error += 1
    random.seed(1)
    random.shuffle(list_ids)
    print(f"Good: {good/(good+scratch+dent+error)}")
    print(f"Scratch: {scratch / (good + scratch + dent + error)}")
    print(f"Dent: {dent / (good + scratch + dent + error)}")
    print(f"Error: {error / (good + scratch + dent + error)}")
    return list_ids


def load_lightfield_data(list_ids, img_size):
    """
    Loads lightfield images from directory.
    Images are loaded in following pattern:
             12
             11
             10
    06 05 04 03 02 01 00
             09
             08
             07

    :param img_size:
    :param list_ids: Paths to directories
    :return: features: (#LF, resX, resY, hor_or_vert, #img, RGB)
             labels: (#LF)
    """
    # print("\nNow training on:")
    features = np.zeros((len(list_ids), img_size, img_size, 14), np.float32)
    labels = np.zeros((len(list_ids)), np.int64)
    for i, lf in enumerate(list_ids):
        # print(lf.split('\\')[-2:])

        with Image.open(f"{lf}\\0002_Set0_Cam_003_img.png") as im:
            im_resize = np.array(im.resize((img_size, img_size))).astype('float32')
        for k in range(14):
            features[i, :, :, k] = (0.299*im_resize[:, :, 0] + 0.587*im_resize[:, :, 1]
                                    + 0.114*im_resize[:, :, 2])

        # 0: no defect, 1: scratch, 2: dent
        if 'good' in lf:
            gt = 0
        elif 'scratch' in lf:
            gt = 1
        elif 'dent' in lf:
            gt = 2
        else:
            gt = 'error'
        labels[i] = gt
    return features, labels
