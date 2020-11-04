from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Flatten, Dense, Conv2D
from tensorflow.keras.layers import Dropout, BatchNormalization, Input, MaxPool2D
from tensorflow.keras.backend import concatenate
import tensorflow as tf


def layer1_multistream(res_x, res_y, num_cams, filter_num):
    """
    Multi-stream layer: Conv - ReLU - Conv - ReLU - BN

    :param res_x:
    :param res_y:
    :param num_cams:
    :param filter_num:
    :return:
    """
    if not hasattr(layer1_multistream, "instance"):
        layer1_multistream.instance = 0
    j = layer1_multistream.instance
    seq = Sequential()
    for i in range(3):
        seq.add(Conv2D(filter_num, (2, 2), input_shape=(res_x, res_y, num_cams),
                       padding='same', name=f'S1_C1{i}_{j}', activation='relu'))
        seq.add(Conv2D(filter_num, (2, 2), padding='same',
                       name=f'S1_C2{i}_{j}', activation='relu'))
        # In original activation comes after BN, but other way round may be better:
        # https://blog.paperspace.com/busting-the-myths-about-batch-normalization/
        seq.add(BatchNormalization(axis=-1, name=f'S1_BN{i}_{j}'))
    # seq.add(Reshape(input_dim1-6, input_dim2-6, filter_num))
    layer1_multistream.instance += 1
    return seq


def layer2_merged(res_x, res_y, filter_num, conv_depth):
    """
    Merged layer: Conv - ReLU - Conv - ReLU - BN

    :param res_x:
    :param res_y:
    :param filter_num: twice that of layer 1 (2x70)
    :param conv_depth: should be 6 blocks
    :return:
    """
    seq = Sequential()
    for i in range(conv_depth):
        # seq.add(MaxPool2D((2, 2), name=f'S2_MP{i}'))  # v Do strides instead of MaxPooling? v
        seq.add(Conv2D(filter_num, (2, 2), strides=(2, 2),
                       input_shape=(int(res_x/2**i), int(res_y/2**i), filter_num),
                       padding='valid', name=f'S2_C1{i}', activation='relu'))
        seq.add(Conv2D(filter_num, (2, 2), padding='same', name=f'S2_C2{i}', activation='relu'))
        seq.add(BatchNormalization(axis=-1, name=f'S2_BN{i}'))
    return seq


def layer3_last():
    """
    Last layer: Flatten - Dense - ReLU - Dense - Sigmoid

    :return:
    """
    seq = Sequential()
    seq.add(Flatten())
    seq.add(Dense(128, activation='relu', name=f"S3_d1"))
    seq.add(Dense(64, activation='relu', name=f"S3_d2"))
    seq.add(Dense(3, activation='sigmoid', name=f"S3_dfinal"))
    return seq


def define_epidef(sz_input1, sz_input2, view_n, conv_depth, filter_num):
    """
    Compiles the full network.

    :param sz_input1: resX
    :param sz_input2: resY
    :param view_n: num_cams
    :param conv_depth: number of blocks in second layer
    :param filter_num: number of channels in multistream layers
    :return:
    """
    # 2-Input: Conv - ReLU - Conv - ReLU - BN
    input_stack_vert = Input(shape=(sz_input1, sz_input2, view_n), name='input_stack_vert')
    input_stack_hori = Input(shape=(sz_input1, sz_input2, view_n), name='input_stack_hori')

    # 2-Stream layer: Conv - ReLU - Conv - ReLU - BN
    mid_vert = layer1_multistream(sz_input1, sz_input2, view_n, filter_num)(input_stack_vert)
    mid_hori = layer1_multistream(sz_input1, sz_input2, view_n, filter_num)(input_stack_hori)

    # Merge layers
    mid_merged = concatenate([mid_vert, mid_hori])

    # Merged layer: MaxPool - Conv - ReLU - Conv - ReLU - BN
    mid_merged_ = layer2_merged(sz_input1, sz_input2, 2*filter_num, conv_depth)(mid_merged)

    # Last Dense layer: Dense - ReLU - Dense
    output = layer3_last()(mid_merged_)
    model_512 = Model(inputs=[input_stack_vert, input_stack_hori], outputs=[output])
    METRICS = ['accuracy',
               tf.keras.metrics.Precision(name='precision'),
               tf.keras.metrics.Recall(name='recall')]
    model_512.compile(loss='categorical_crossentropy', optimizer='adam', metrics=METRICS)
    model_512.summary()
    return model_512
