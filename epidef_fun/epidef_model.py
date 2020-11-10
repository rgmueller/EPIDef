from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Flatten, Dense, Conv2D
from tensorflow.keras.layers import Dropout, BatchNormalization, Input, MaxPool2D, DepthwiseConv2D
from tensorflow.keras.layers import Add
from tensorflow.keras.backend import concatenate
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
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


# def get_top(x_input):
#     """Block top operations
#     This functions apply Batch Normalization and Leaky ReLU activation to the input.
#     # Arguments:
#         x_input: Tensor, input to apply BN and activation  to.
#     # Returns:
#         Output tensor
#     """
#
#     x = tf.keras.layers.BatchNormalization()(x_input)
#     x = tf.keras.layers.LeakyReLU()(x)
#     return x


# def get_block(x_input, input_channels, output_channels):
#     """MBConv block
#     This function defines a mobile Inverted Residual Bottleneck block with BN and Leaky ReLU
#     # Arguments
#         x_input: Tensor, input tensor of conv layer.
#         input_channels: Integer, the dimentionality of the input space.
#         output_channels: Integer, the dimensionality of the output space.
#
#     # Returns
#         Output tensor.
#     """
#
#     x = tf.keras.layers.Conv2D(input_channels, kernel_size=(1, 1), padding='same', use_bias=False)(
#         x_input)
#     x = get_top(x)
#     x = tf.keras.layers.DepthwiseConv2D(kernel_size=(1, 3), padding='same', use_bias=False)(x)
#     x = get_top(x)
#     x = tf.keras.layers.MaxPooling2D(pool_size=(2, 1), strides=(2, 1))(x)
#     x = tf.keras.layers.DepthwiseConv2D(kernel_size=(3, 1), padding='same', use_bias=False)(x)
#     x = get_top(x)
#     x = tf.keras.layers.Conv2D(output_channels, kernel_size=(2, 1), strides=(1, 2), padding='same',
#                                use_bias=False)(x)
#     return x


# def eff_net(input_shape, num_classes, plot_model=False):
#     """EffNet
#     This function defines a EfficientNet architecture.
#     # Arguments
#         input_shape: An integer or tuple/list of 3 integers, shape
#             of input tensor.
#         num_classes: Integer, number of classes.
#         plot_model: Boolean, whether to plot model architecture or not
#     # Returns
#         EfficientNet model.
#     """
#     x_input = tf.keras.layers.Input(shape=input_shape)
#     x = get_block(x_input, 32, 64)
#     x = get_block(x, 64, 128)
#     x = get_block(x, 128, 256)
#     x = tf.keras.layers.Flatten()(x)
#     x = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
#     model = tf.keras.models.Model(inputs=x_input, outputs=x)
#
#     if plot_model:
#         tf.keras.utils.plot_model(model, to_file='model.png', show_shapes=True)
#
#     return model


def inverted_residual_block(x, expand=64, squeeze=16):
    m = Conv2D(expand, (1, 1), activation='relu')(x)
    m = DepthwiseConv2D((3, 3), activation='relu')(m)
    m = Conv2D(squeeze, (1, 1), activation='relu')(m)
    return Add()([m, x])


def bottleneck_block(x, stride, expand=64, squeeze=16):
    m = Conv2D(expand, (1, 1), strides=(stride, stride), activation='relu')(x)
    m = BatchNormalization()(m)
    m = DepthwiseConv2D((3, 3), activation='relu')(m)
    m = BatchNormalization()(m)
    m = Conv2D(squeeze, (1, 1))(m)  # no activation on last convolution
    m = BatchNormalization()(m)
    return Add()([m, x])


def efficientnet():
    """
    Merged layer: Conv - ReLU - Conv - ReLU - BN

    :param res_x:
    :param res_y:
    :param filter_num: twice that of layer 1 (2x70)
    :param conv_depth: should be 6 blocks
    :return: seq:
    """
    # seq = EffNet((224, 224, 140), 3)
    seq = efficientnet_model.EfficientNet()
    # seq = Sequential()
    # seq.add(Conv2D(140, (3, 3), strides=(2, 2), padding='same', activation='relu'))
    # seq.add(BatchNormalization(axis=-1))
    #
    # seq.add(Flatten())
    # seq.add(Dense(3, activation='softmax'))
    # seq.add(EfficientNetB0(include_top=True, weights=None, classes=3))
    return seq


# def layer3_last():
#     """
#     Last layer: Flatten - Dense - ReLU - Dense - Sigmoid
#
#     :return:
#     """
#     seq = Sequential()
#     seq.add(Flatten())
#     seq.add(Dense(256, activation='relu', name=f"S3_d1"))
#     seq.add(Dense(128, activation='relu', name=f"S3_d2"))
#     seq.add(Dense(3, activation='softmax', name=f"S3_dfinal"))
#     return seq


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
