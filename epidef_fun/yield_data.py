def load_lightfield_data(lf_directory):
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

    :param lf_directory: Path to directory
    :return: features: (#LF, resX, resY, hor_or_vert, #img, RGB)
             labels: (#LF)
             statistics: number of good/scratch/defect for each model (#model, good/scratch/dent)
    """
    image_dirs = []
    for path, subdirs, files in os.walk(lf_directory):
        if '0002_Set0_Cam_003_img.png' in files:
            image_dirs.append(path)
    random.seed(1)
    random.shuffle(image_dirs)
    features = np.zeros((len(image_dirs[:250]), 400, 400, 2, 7, 3), np.float32)
    labels = np.zeros((len(image_dirs[:250])), np.int64)
    for i, lf in enumerate(image_dirs[:250]):
        print(lf)
        tmp = np.float32(imageio.imread(f"{lf}\\0002_Set0_Cam_000_img.png"))
        features[i, :, :, 0, 6, :] = tmp[:, :, :3]  # rightmost image
        tmp = np.float32(imageio.imread(f"{lf}\\0002_Set0_Cam_001_img.png"))
        features[i, :, :, 0, 5, :] = tmp[:, :, :3]  # (R,G,B,alpha)
        tmp = np.float32(imageio.imread(f"{lf}\\0002_Set0_Cam_002_img.png"))
        features[i, :, :, 0, 4, :] = tmp[:, :, :3]
        tmp = np.float32(imageio.imread(f"{lf}\\0002_Set0_Cam_003_img.png"))
        features[i, :, :, 0, 3, :] = tmp[:, :, :3]  # center image
        features[i, :, :, 1, 3, :] = tmp[:, :, :3]
        tmp = np.float32(imageio.imread(f"{lf}\\0002_Set0_Cam_004_img.png"))
        features[i, :, :, 0, 2, :] = tmp[:, :, :3]
        tmp = np.float32(imageio.imread(f"{lf}\\0002_Set0_Cam_005_img.png"))
        features[i, :, :, 0, 1, :] = tmp[:, :, :3]
        tmp = np.float32(imageio.imread(f"{lf}\\0002_Set0_Cam_006_img.png"))
        features[i, :, :, 0, 0, :] = tmp[:, :, :3]  # leftmost image
        tmp = np.float32(imageio.imread(f"{lf}\\0002_Set0_Cam_007_img.png"))
        features[i, :, :, 1, 0, :] = tmp[:, :, :3]  # bottom image
        tmp = np.float32(imageio.imread(f"{lf}\\0002_Set0_Cam_008_img.png"))
        features[i, :, :, 1, 1, :] = tmp[:, :, :3]
        tmp = np.float32(imageio.imread(f"{lf}\\0002_Set0_Cam_009_img.png"))
        features[i, :, :, 1, 2, :] = tmp[:, :, :3]
        tmp = np.float32(imageio.imread(f"{lf}\\0002_Set0_Cam_010_img.png"))
        features[i, :, :, 1, 4, :] = tmp[:, :, :3]
        tmp = np.float32(imageio.imread(f"{lf}\\0002_Set0_Cam_011_img.png"))
        features[i, :, :, 1, 5, :] = tmp[:, :, :3]
        tmp = np.float32(imageio.imread(f"{lf}\\0002_Set0_Cam_012_img.png"))
        features[i, :, :, 1, 6, :] = tmp[:, :, :3]  # top image
        del tmp

        # xxxx_mxdefect
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