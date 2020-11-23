import numpy as np


def generate_traindata(x, y, input_size, batch_size, num_cams):
    """
    Generates training data using LF images and disparity maps by randomly chosen variables.

    :param x: (#LF, resX, resY, hori_or_vert, num_cams, RGB)
    :param y: (#LF)
    :param input_size: resX/resY
    :param batch_size: size of batch
    :param num_cams: number of cameras along one direction
    :return: x_hori: (batch_size, rexX, resY, num_cams)
             x_vert: (batch_size, rexX, resY, num_cams)
             label: (batch_size)
             
    1. Gray image: random R,G,B --> R*img_R + G*img_G + B*img_B
    2. patch-wise learning: random x,y  --> LFimage[x:x+size1,y:y+size2]
    3. scale augmentation: scale 1,2,3  --> ex> LFimage[x:x+2*size1:2,y:y+2*size2:2]
    (not sure if 2. and 3. are applicable for defect detection)
    """

    # Initialize image stack and labels
    res_x = input_size
    res_y = input_size
    x = np.zeros((batch_size, res_x, res_y, 14), dtype=np.float32)
    x = x/255

    return x, y


def data_augmentation(x, traindata_labels, batch_size, train=True):
    """
    Performs data augmentation. (Rotation, transpose, gamma)

    :param x: (batch_size, resX, rexY, num_cams)
    :param traindata_labels: (batch_size)
    :param batch_size: size of batch
    :param train: turns on data augmentation
    :return: x_vert: (batch_size, resX, rexY, num_cams)
             x_hori: (batch_size, resX, rexY, num_cams)
             traindata_labels: (batch_size)
    """
    for batch_i in range(batch_size):
        gray_rand = 0.4 * np.random.rand()+0.8

        x[batch_i, :, :, :] = pow(x[batch_i, :, :, :], gray_rand)

        roll = np.random.randint(-12, 13)
        if train:
            translate = np.random.randint(0, 4)
        else:
            translate = 0
        if translate == 1:  # translate x-direction
            x_tmp = np.copy(np.roll(x[batch_i, :, :, :], roll, axis=0))
            x[batch_i, :, :, :] = x_tmp

        if translate == 2:  # translate y-direction
            x_tmp = np.copy(np.roll(x[batch_i, :, :, :], roll, axis=1))
            x[batch_i, :, :, :] = x_tmp

        if translate == 3:  # translate diagonally
            x_tmp = np.copy(np.roll(x[batch_i, :, :, :], roll, axis=(0, 1)))
            x[batch_i, :, :, :] = x_tmp

        if train:
            rotation_or_transpose = np.random.randint(0, 6)
        else:
            rotation_or_transpose = 0
        if rotation_or_transpose == 4:  # Transpose
            x_tmp = np.copy(np.transpose(np.squeeze(x[batch_i, :, :, :]), (1, 0, 2)))
            x[batch_i, :, :, :] = np.copy(x_tmp[:, :, ::-1])

        if rotation_or_transpose == 1:  # 90 degrees
            x_tmp = np.copy(np.rot90(x[batch_i, :, :, :], 1, (0, 1)))
            x[batch_i, :, :, :] = x_tmp

        if rotation_or_transpose == 2:  # 180 degrees
            x_tmp = np.copy(np.rot90(x[batch_i, :, :, :], 2, (0, 1)))
            x[batch_i, :, :, :] = x_tmp[:, :, ::-1]

        if rotation_or_transpose == 3:  # 270 degrees
            x_tmp = np.copy(np.rot90(x[batch_i, :, :, :], 3, (0, 1)))
            x[batch_i, :, :, :] = x_tmp[:, :, ::-1]

    return x, traindata_labels
