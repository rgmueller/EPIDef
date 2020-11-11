from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Flatten, Dense, Conv2D
from tensorflow.keras.layers import Dropout, BatchNormalization, Input, MaxPool2D, Add
from tensorflow.keras.backend import concatenate
import tensorflow as tf
from official.vision.image_classification.efficientnet import efficientnet_model


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
    seq.add(Conv2D(filter_num, (3, 3), input_shape=(res_x, res_y, num_cams),
                   padding='valid', name=f'S1_C10_{j}', activation='relu'))
    seq.add(Conv2D(filter_num, (3, 3), input_shape=(res_x-2, res_y-2, num_cams),
                   padding='valid', name=f'S1_C20_{j}', activation='relu'))
    # In original activation comes after BN, but other way round may be better:
    # https://blog.paperspace.com/busting-the-myths-about-batch-normalization/
    seq.add(BatchNormalization(axis=-1, name=f'S1_BN0_{j}'))

    seq.add(Conv2D(filter_num, (3, 3), input_shape=(res_x-4, res_y-4, num_cams),
                   padding='valid', name=f'S1_C11_{j}', activation='relu'))
    seq.add(Conv2D(filter_num, (3, 3), input_shape=(res_x-6, res_y-6, num_cams),
                   padding='valid', name=f'S1_C21_{j}', activation='relu'))
    seq.add(BatchNormalization(axis=-1, name=f'S1_BN1_{j}'))

    seq.add(Conv2D(filter_num, (3, 3), input_shape=(res_x-8, res_y-8, num_cams),
                   padding='valid', name=f'S1_C12_{j}', activation='relu'))
    seq.add(Conv2D(filter_num, (3, 3), input_shape=(res_x-10, res_y-10, num_cams),
                   padding='valid', name=f'S1_C22_{j}', activation='relu'))
    seq.add(BatchNormalization(axis=-1, name=f'S1_BN2_{j}'))
    layer1_multistream.instance += 1
    return seq


def efficientnet():
    """
    Merged layer: Conv - ReLU - Conv - ReLU - BN

    :return: seq:
    """
    seq = efficientnet_model.EfficientNet()

    return seq


def define_epidef(sz_input1, sz_input2, view_n, filter_num):
    """
    Compiles the full network.

    :param sz_input1: resX
    :param sz_input2: resY
    :param view_n: num_cams
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
    # mid_merged_ = layer2_merged(sz_input1, sz_input2, 2*filter_num, conv_depth)(mid_merged)
    mid_merged_ = efficientnet()

    output = mid_merged_(mid_merged)
    model_512 = Model(inputs=[input_stack_vert, input_stack_hori], outputs=[output])
    metrics = ['accuracy',
               tf.keras.metrics.Precision(name='precision'),
               tf.keras.metrics.Recall(name='recall')]
    model_512.compile(loss='categorical_crossentropy', optimizer='adam', metrics=metrics)
    model_512.summary()
    return model_512
