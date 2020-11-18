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
    x_vert = np.zeros((batch_size, res_x, res_y, num_cams), dtype=np.float32)
    x_hori = np.zeros((batch_size, res_x, res_y, num_cams), dtype=np.float32)
    label = np.zeros(batch_size)

    # Generate image stacks
    for i in range(batch_size):
        # Variables for gray conversion
        # rand_3color = 0.05 + np.random.rand(3)
        # rand_3color = rand_3color/np.sum(rand_3color)
        # r = rand_3color[0]
        # g = rand_3color[1]
        # b = rand_3color[2]
        r = 0.299
        g = 0.587
        b = 0.114

        # Since we always use 7x7 images the center view stays the same
        # Two image stacks are selected and gray-scaled
        x_vert[i, :, :, :] = (r * x[i, :, :, 1, :, 0]
                              + g * x[i, :, :, 1, :, 1]
                              + b * x[i, :, :, 1, :, 2]).astype('float32')
        x_hori[i, :, :, :] = (r * x[i, :, :, 0, :, 0]
                              + g * x[i, :, :, 0, :, 1]
                              + b * x[i, :, :, 0, :, 2]).astype('float32')
        label[i] = y[i]

    x_hori = x_hori/255
    x_vert = x_vert/255
    return x_vert, x_hori, label


def data_augmentation(x_vert, x_hori, traindata_labels, batch_size, train=True):
    """
    Performs data augmentation. (Rotation, transpose, gamma)

    :param x_hori: (batch_size, resX, rexY, num_cams)
    :param x_vert: (batch_size, resX, rexY, num_cams)
    :param traindata_labels: (batch_size)
    :param batch_size: size of batch
    :param train: turns on data augmentation
    :return: x_vert: (batch_size, resX, rexY, num_cams)
             x_hori: (batch_size, resX, rexY, num_cams)
             traindata_labels: (batch_size)
    """
    for batch_i in range(batch_size):
        gray_rand = 0.4 * np.random.rand()+0.8

        x_hori[batch_i, :, :, :] = pow(x_hori[batch_i, :, :, :], gray_rand)
        x_vert[batch_i, :, :, :] = pow(x_vert[batch_i, :, :, :], gray_rand)

        roll = np.random.randint(-12, 13)
        if train:
            translate = np.random.randint(0, 4)
        else:
            translate = 0
        if translate == 1:  # translate x-direction
            x_vert_tmp = np.copy(np.roll(x_vert[batch_i, :, :, :], roll, axis=1))
            x_hori_tmp = np.copy(np.roll(x_hori[batch_i, :, :, :], roll, axis=1))
            x_vert[batch_i, :, :, :] = x_vert_tmp
            x_hori[batch_i, :, :, :] = x_hori_tmp

        if translate == 2:  # translate y-direction
            x_vert_tmp = np.copy(np.roll(x_vert[batch_i, :, :, :], roll, axis=0))
            x_hori_tmp = np.copy(np.roll(x_hori[batch_i, :, :, :], roll, axis=0))
            x_vert[batch_i, :, :, :] = x_vert_tmp
            x_hori[batch_i, :, :, :] = x_hori_tmp

        if translate == 3:  # translate diagonally
            x_vert_tmp = np.copy(np.roll(x_vert[batch_i, :, :, :], roll, axis=(0, 1)))
            x_hori_tmp = np.copy(np.roll(x_hori[batch_i, :, :, :], roll, axis=(0, 1)))
            x_vert[batch_i, :, :, :] = x_vert_tmp
            x_hori[batch_i, :, :, :] = x_hori_tmp


        if train:
            rotation_or_transpose = np.random.randint(0, 6)
        else:
            rotation_or_transpose = 0
        if rotation_or_transpose == 4:  # Transpose
            x_hori_tmp = np.copy(np.transpose(np.squeeze(x_hori[batch_i, :, :, :]), (1, 0, 2)))
            x_vert_tmp = np.copy(np.transpose(np.squeeze(x_vert[batch_i, :, :, :]), (1, 0, 2)))
            x_hori[batch_i, :, :, :] = np.copy(x_vert_tmp[:, :, ::-1])
            x_vert[batch_i, :, :, :] = np.copy(x_hori_tmp[:, :, ::-1])

        if rotation_or_transpose == 1:  # 90 degrees
            x_hori_tmp = np.copy(np.rot90(x_hori[batch_i, :, :, :], 1, (0, 1)))
            x_vert_tmp = np.copy(np.rot90(x_vert[batch_i, :, :, :], 1, (0, 1)))
            x_vert[batch_i, :, :, :] = x_hori_tmp
            x_hori[batch_i, :, :, :] = x_vert_tmp

        if rotation_or_transpose == 2:  # 180 degrees
            x_hori_tmp = np.copy(np.rot90(x_hori[batch_i, :, :, :], 2, (0, 1)))
            x_vert_tmp = np.copy(np.rot90(x_vert[batch_i, :, :, :], 2, (0, 1)))
            x_vert[batch_i, :, :, :] = x_vert_tmp[:, :, ::-1]
            x_hori[batch_i, :, :, :] = x_hori_tmp[:, :, ::-1]

        if rotation_or_transpose == 3:  # 270 degrees
            x_hori_tmp = np.copy(np.rot90(x_hori[batch_i, :, :, :], 3, (0, 1)))
            x_vert_tmp = np.copy(np.rot90(x_vert[batch_i, :, :, :], 3, (0, 1)))
            x_vert[batch_i, :, :, :] = x_hori_tmp[:, :, ::-1]
            x_hori[batch_i, :, :, :] = x_vert_tmp



    return x_vert, x_hori, traindata_labels


def generate_testdata(x, y, num_cams):
    input_size = 512
    testdata_vert = np.zeros((len(y), input_size, input_size, num_cams), dtype=np.float32)
    testdata_hori = np.zeros((len(y), input_size, input_size, num_cams), dtype=np.float32)
    testdata_batch_label = np.zeros(len(y))

    for ii in range(len(y)):
        # These are the values from the paper, not sure why they were chosen like this
        r = 0.299
        g = 0.587
        b = 0.114

        image_id = ii
        testdata_vert[ii, :, :, :] = (r * x[image_id, :, :, 1, :, 0]
                                      + g * x[image_id, :, :, 1, :, 1]
                                      + b * x[image_id, :, :, 1, :, 2]).astype('float32')
        testdata_hori[ii, :, :, :] = (r * x[image_id, :, :, 0, :, 0]
                                      + g * x[image_id, :, :, 0, :, 1]
                                      + b * x[image_id, :, :, 0, :, 2]).astype('float32')
        testdata_batch_label[ii] = y[image_id]

    testdata_hori = testdata_hori/255
    testdata_vert = testdata_vert/255

    return testdata_vert, testdata_hori, testdata_batch_label
