import numpy as np


def generate_traindata(traindata_all, traindata_label, input_size, batch_size, num_cams):
    """
    Generates training data using LF images and disparity maps by randomly chosen variables.

    :param traindata_all: (#LF, resX, resY, hori_or_vert, num_cams, RGB)
    :param traindata_label: (#LF)
    :param input_size: resX/resY
    :param batch_size: size of batch
    :param num_cams: number of cameras along one direction
    :return: x_hori: (batch_size, rexX, resY, num_cams)
             x_vert: (batch_size, rexX, resY, num_cams)
             traindata_batch_label: (batch_size)
             
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
    traindata_batch_label = np.zeros(batch_size)

    # Generate image stacks
    for ii in range(0, batch_size):
        # Variables for gray conversion
        # rand_3color = 0.05 + np.random.rand(3)
        # rand_3color = rand_3color/np.sum(rand_3color)
        # r = rand_3color[0]
        # g = rand_3color[1]
        # b = rand_3color[2]
        r = 0.299
        g = 0.587
        b = 0.114

        # choose one random lightfield out of training data
        image_id = np.random.choice(np.arange(traindata_all.shape[0]))

        # Since we always use 7x7 images the center view stays the same
        # Two image stacks are selected and gray-scaled
        x_vert[ii, :, :, :] = (r*traindata_all[image_id, :, :, 1, :, 0]
                               + g*traindata_all[image_id, :, :, 1, :, 1]
                               + b*traindata_all[image_id, :, :, 1, :, 2]).astype('float32')
        x_hori[ii, :, :, :] = (r*traindata_all[image_id, :, :, 0, :, 0]
                               + g*traindata_all[image_id, :, :, 0, :, 1]
                               + b*traindata_all[image_id, :, :, 0, :, 2]).astype('float32')
        traindata_batch_label[ii] = traindata_label[image_id]

    x_hori = x_hori/255
    x_vert = x_vert/255

    return x_hori, x_vert, traindata_batch_label


def data_augmentation(traindata_hori, traindata_vert, traindata_labels, batch_size):
    """
    Performs data augmentation. (Rotation, transpose, gamma)

    :param traindata_hori: (batch_size, resX, rexY, num_cams)
    :param traindata_vert: (batch_size, resX, rexY, num_cams)
    :param traindata_labels: (batch_size)
    :param batch_size: size of batch
    :return: traindata_hori: (batch_size, resX, rexY, num_cams)
             traindata_vert: (batch_size, resX, rexY, num_cams)
             traindata_labels: (batch_size)
    """
    for batch_i in range(batch_size):
        gray_rand = 0.4 * np.random.rand()+0.8

        traindata_hori[batch_i, :, :, :] = pow(traindata_hori[batch_i, :, :, :], gray_rand)
        traindata_vert[batch_i, :, :, :] = pow(traindata_vert[batch_i, :, :, :], gray_rand)

        rotation_or_transpose = np.random.randint(0, 5)
        if rotation_or_transpose == 4:  # Transpose
            traindata_hori_tmp = np.copy(np.transpose(np.squeeze(
                traindata_hori[batch_i, :, :, :]), (1, 0, 2)))
            traindata_vert_tmp = np.copy(np.transpose(np.squeeze(
                                                      traindata_vert[batch_i, :, :, :]), (1, 0, 2)))
            traindata_hori[batch_i, :, :, :] = np.copy(traindata_vert_tmp[:, :, ::-1])
            traindata_vert[batch_i, :, :, :] = np.copy(traindata_hori_tmp[:, :, ::-1])

        if rotation_or_transpose == 1:  # 90 degrees
            traindata_hori_tmp = np.copy(np.rot90(traindata_hori[batch_i, :, :, :], 1, (0, 1)))
            traindata_vert_tmp = np.copy(np.rot90(traindata_vert[batch_i, :, :, :], 1, (0, 1)))
            traindata_vert[batch_i, :, :, :] = traindata_hori_tmp
            traindata_hori[batch_i, :, :, :] = traindata_vert_tmp

        if rotation_or_transpose == 2:  # 180 degrees
            traindata_hori_tmp = np.copy(np.rot90(traindata_hori[batch_i, :, :, :], 2, (0, 1)))
            traindata_vert_tmp = np.copy(np.rot90(traindata_vert[batch_i, :, :, :], 2, (0, 1)))
            traindata_vert[batch_i, :, :, :] = traindata_vert_tmp[:, :, ::-1]
            traindata_hori[batch_i, :, :, :] = traindata_hori_tmp[:, :, ::-1]

        if rotation_or_transpose == 3:  # 270 degrees
            traindata_hori_tmp = np.copy(np.rot90(traindata_hori[batch_i, :, :, :], 3, (0, 1)))
            traindata_vert_tmp = np.copy(np.rot90(traindata_vert[batch_i, :, :, :], 3, (0, 1)))
            traindata_vert[batch_i, :, :, :] = traindata_hori_tmp[:, :, ::-1]
            traindata_hori[batch_i, :, :, :] = traindata_vert_tmp
    return traindata_hori, traindata_vert, traindata_labels


def generate_testdata(testdata_all, testdata_label, num_cams):
    input_size = 512
    testdata_vert = np.zeros((len(testdata_label), input_size,
                              input_size, num_cams), dtype=np.float32)
    testdata_hori = np.zeros((len(testdata_label), input_size,
                              input_size, num_cams), dtype=np.float32)
    testdata_batch_label = np.zeros(len(testdata_label))

    for ii in range(len(testdata_label)):
        # These are the values from the paper, not sure why they were chosen like this
        r = 0.299
        g = 0.587
        b = 0.114

        image_id = ii
        testdata_vert[ii, :, :, :] = (r * testdata_all[image_id, :, :, 1, :, 0]
                                      + g * testdata_all[image_id, :, :, 1, :, 1]
                                      + b * testdata_all[image_id, :, :, 1, :, 2]).astype('float32')
        testdata_hori[ii, :, :, :] = (r * testdata_all[image_id, :, :, 0, :, 0]
                                      + g * testdata_all[image_id, :, :, 0, :, 1]
                                      + b * testdata_all[image_id, :, :, 0, :, 2]).astype('float32')
        testdata_batch_label[ii] = testdata_label[image_id]

    testdata_hori = testdata_hori/255
    testdata_vert = testdata_vert/255

    return testdata_vert, testdata_hori, testdata_batch_label
