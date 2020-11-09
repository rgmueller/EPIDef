import numpy as np
import imageio
import random
import os
import tensorflow as tf


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

    :param list_IDs: Paths to directories
    :return: features: (#LF, resX, resY, hor_or_vert, #img, RGB)
             labels: (#LF)
             statistics: number of good/scratch/defect for each model (#model, good/scratch/dent)
    """
    print("\nNow training on:")
    features = np.zeros((len(list_IDs), IMG_SIZE, IMG_SIZE, 2, 7, 3), np.float32)
    labels = np.zeros((len(list_IDs)), np.int64)
    for i, lf in enumerate(list_IDs):
        print(lf.split('\\')[-2:])
        tmp = np.float32(imageio.imread(f"{lf}\\0001_Set0_Cam_000_img.png"))
        tmp = tf.image.resize(tmp, (IMG_SIZE, IMG_SIZE))
        features[i, :, :, 0, 6, :] = tmp[:, :, :3]  # rightmost image
        tmp = np.float32(imageio.imread(f"{lf}\\0001_Set0_Cam_001_img.png"))
        tmp = tf.image.resize(tmp, (IMG_SIZE, IMG_SIZE))
        features[i, :, :, 0, 5, :] = tmp[:, :, :3]  # (R,G,B,alpha)
        tmp = np.float32(imageio.imread(f"{lf}\\0001_Set0_Cam_002_img.png"))
        tmp = tf.image.resize(tmp, (IMG_SIZE, IMG_SIZE))
        features[i, :, :, 0, 4, :] = tmp[:, :, :3]
        tmp = np.float32(imageio.imread(f"{lf}\\0001_Set0_Cam_003_img.png"))
        tmp = tf.image.resize(tmp, (IMG_SIZE, IMG_SIZE))
        features[i, :, :, 0, 3, :] = tmp[:, :, :3]  # center image
        features[i, :, :, 1, 3, :] = tmp[:, :, :3]
        tmp = np.float32(imageio.imread(f"{lf}\\0001_Set0_Cam_004_img.png"))
        tmp = tf.image.resize(tmp, (IMG_SIZE, IMG_SIZE))
        features[i, :, :, 0, 2, :] = tmp[:, :, :3]
        tmp = np.float32(imageio.imread(f"{lf}\\0001_Set0_Cam_005_img.png"))
        tmp = tf.image.resize(tmp, (IMG_SIZE, IMG_SIZE))
        features[i, :, :, 0, 1, :] = tmp[:, :, :3]
        tmp = np.float32(imageio.imread(f"{lf}\\0001_Set0_Cam_006_img.png"))
        tmp = tf.image.resize(tmp, (IMG_SIZE, IMG_SIZE))
        features[i, :, :, 0, 0, :] = tmp[:, :, :3]  # leftmost image
        tmp = np.float32(imageio.imread(f"{lf}\\0001_Set0_Cam_007_img.png"))
        tmp = tf.image.resize(tmp, (IMG_SIZE, IMG_SIZE))
        features[i, :, :, 1, 0, :] = tmp[:, :, :3]  # bottom image
        tmp = np.float32(imageio.imread(f"{lf}\\0001_Set0_Cam_008_img.png"))
        tmp = tf.image.resize(tmp, (IMG_SIZE, IMG_SIZE))
        features[i, :, :, 1, 1, :] = tmp[:, :, :3]
        tmp = np.float32(imageio.imread(f"{lf}\\0001_Set0_Cam_009_img.png"))
        tmp = tf.image.resize(tmp, (IMG_SIZE, IMG_SIZE))
        features[i, :, :, 1, 2, :] = tmp[:, :, :3]
        tmp = np.float32(imageio.imread(f"{lf}\\0001_Set0_Cam_010_img.png"))
        tmp = tf.image.resize(tmp, (IMG_SIZE, IMG_SIZE))
        features[i, :, :, 1, 4, :] = tmp[:, :, :3]
        tmp = np.float32(imageio.imread(f"{lf}\\0001_Set0_Cam_011_img.png"))
        tmp = tf.image.resize(tmp, (IMG_SIZE, IMG_SIZE))
        features[i, :, :, 1, 5, :] = tmp[:, :, :3]
        tmp = np.float32(imageio.imread(f"{lf}\\0001_Set0_Cam_012_img.png"))
        tmp = tf.image.resize(tmp, (IMG_SIZE, IMG_SIZE))
        features[i, :, :, 1, 6, :] = tmp[:, :, :3]  # top image
        del tmp

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
