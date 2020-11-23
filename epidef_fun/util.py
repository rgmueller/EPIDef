import numpy as np
import random
import os
from PIL import Image


def get_list_IDs(lf_directory):
    """

    :param lf_directory:
    :return: list_IDs:
    """
    good = 0
    scratch = 0
    dent = 0
    error = 0
    list_IDs = []
    for path, subdirs, files in os.walk(lf_directory):
        if '0001_Set0_Cam_003_img.png' in files:  # only add paths with images
            list_IDs.append(path)
            if 'good' in path:
                good += 1
            elif 'scratch' in path:
                scratch += 1
            elif 'dent' in path:
                dent += 1
            else:
                error += 1
    random.seed(1)
    random.shuffle(list_IDs)
    print(f"Good: {good/(good+scratch+dent+error)}")
    print(f"Scratch: {scratch / (good + scratch + dent + error)}")
    print(f"Dent: {dent / (good + scratch + dent + error)}")
    print(f"Error: {error / (good + scratch + dent + error)}")
    return list_IDs


def load_lightfield_data(list_IDs, IMG_SIZE):
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

    :param IMG_SIZE:
    :param list_IDs: Paths to directories
    :return: features: (#LF, resX, resY, hor_or_vert, #img, RGB)
             labels: (#LF)
    """
    # print("\nNow training on:")
    features = np.zeros((len(list_IDs), IMG_SIZE, IMG_SIZE, 2, 7, 3), np.float32)
    labels = np.zeros((len(list_IDs)), np.int64)
    for i, lf in enumerate(list_IDs):
        # print(lf.split('\\')[-2:])
        with Image.open(f"{lf}\\0001_Set0_Cam_000_img.png") as im:
            im_resize = np.array(im.resize((IMG_SIZE, IMG_SIZE))).astype('float32')
        features[i, :, :, 0, 6, :] = im_resize[:, :, :3]  # rightmost image
        with Image.open(f"{lf}\\0001_Set0_Cam_001_img.png") as im:
            im_resize = np.array(im.resize((IMG_SIZE, IMG_SIZE))).astype('float32')
        features[i, :, :, 0, 5, :] = im_resize[:, :, :3]  # (R,G,B,alpha)
        with Image.open(f"{lf}\\0001_Set0_Cam_002_img.png") as im:
            im_resize = np.array(im.resize((IMG_SIZE, IMG_SIZE))).astype('float32')
        features[i, :, :, 0, 4, :] = im_resize[:, :, :3]
        with Image.open(f"{lf}\\0001_Set0_Cam_003_img.png") as im:
            im_resize = np.array(im.resize((IMG_SIZE, IMG_SIZE))).astype('float32')
        features[i, :, :, 0, 3, :] = im_resize[:, :, :3]  # center image
        features[i, :, :, 1, 3, :] = im_resize[:, :, :3]
        with Image.open(f"{lf}\\0001_Set0_Cam_004_img.png") as im:
            im_resize = np.array(im.resize((IMG_SIZE, IMG_SIZE))).astype('float32')
        features[i, :, :, 0, 2, :] = im_resize[:, :, :3]
        with Image.open(f"{lf}\\0001_Set0_Cam_005_img.png") as im:
            im_resize = np.array(im.resize((IMG_SIZE, IMG_SIZE))).astype('float32')
        features[i, :, :, 0, 1, :] = im_resize[:, :, :3]
        with Image.open(f"{lf}\\0001_Set0_Cam_006_img.png") as im:
            im_resize = np.array(im.resize((IMG_SIZE, IMG_SIZE))).astype('float32')
        features[i, :, :, 0, 0, :] = im_resize[:, :, :3]  # leftmost image
        with Image.open(f"{lf}\\0001_Set0_Cam_007_img.png") as im:
            im_resize = np.array(im.resize((IMG_SIZE, IMG_SIZE))).astype('float32')
        features[i, :, :, 1, 0, :] = im_resize[:, :, :3]  # bottom image
        with Image.open(f"{lf}\\0001_Set0_Cam_008_img.png") as im:
            im_resize = np.array(im.resize((IMG_SIZE, IMG_SIZE))).astype('float32')
        features[i, :, :, 1, 1, :] = im_resize[:, :, :3]
        with Image.open(f"{lf}\\0001_Set0_Cam_009_img.png") as im:
            im_resize = np.array(im.resize((IMG_SIZE, IMG_SIZE))).astype('float32')
        features[i, :, :, 1, 2, :] = im_resize[:, :, :3]
        with Image.open(f"{lf}\\0001_Set0_Cam_010_img.png") as im:
            im_resize = np.array(im.resize((IMG_SIZE, IMG_SIZE))).astype('float32')
        features[i, :, :, 1, 4, :] = im_resize[:, :, :3]
        with Image.open(f"{lf}\\0001_Set0_Cam_011_img.png") as im:
            im_resize = np.array(im.resize((IMG_SIZE, IMG_SIZE))).astype('float32')
        features[i, :, :, 1, 5, :] = im_resize[:, :, :3]
        with Image.open(f"{lf}\\0001_Set0_Cam_012_img.png") as im:
            im_resize = np.array(im.resize((IMG_SIZE, IMG_SIZE))).astype('float32')
        features[i, :, :, 1, 6, :] = im_resize[:, :, :3]  # top image

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
