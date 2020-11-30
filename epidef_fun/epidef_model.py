from tensorflow.keras import Model
from tensorflow.keras.layers import Input
import tensorflow as tf
from official.vision.image_classification.efficientnet import efficientnet_model


def efficientnet():
    """
    Merged layer: Conv - ReLU - Conv - ReLU - BN

    :return: seq:
    """
    block_config = efficientnet_model.BlockConfig()

    blocks = (  # (input_filters, output_filters, kernel_size, num_repeat,
        #  expand_ratio, strides, se_ratio)
        block_config.from_args(32, 16, 3, 1, 1, (1, 1), 0.25),
        block_config.from_args(16, 24, 3, 2, 6, (2, 2), 0.25),
        block_config.from_args(24, 40, 5, 2, 6, (2, 2), 0.25),
        block_config.from_args(40, 80, 3, 3, 6, (2, 2), 0.25),
        block_config.from_args(80, 112, 5, 3, 6, (1, 1), 0.25),
        block_config.from_args(112, 192, 5, 4, 6, (2, 2), 0.25),
        block_config.from_args(192, 320, 3, 1, 6, (1, 1), 0.25),
    )
    seq = efficientnet_model.EfficientNet(overrides={'num_classes': 3,
                                                     'input_channels': 14,
                                                     'rescale_input': False,
                                                     'blocks': blocks,
                                                     'stem_base_filters': 32})
    return seq


def define_epidef(sz_input1, sz_input2):
    """
    Compiles the full network.

    :param sz_input1: resX
    :param sz_input2: resY
    :return:
    """
    # 2-Input: Conv - ReLU - Conv - ReLU - BN
    input_ = Input(shape=(sz_input1, sz_input2, 14), name='input_stack_vert')

    # Merge layers
    mid_merged_ = efficientnet()

    output = mid_merged_(input_)
    model_512 = Model(inputs=input_, outputs=[output])
    metrics = ['accuracy',
               tf.keras.metrics.Precision(name='precision'),
               tf.keras.metrics.Recall(name='recall')]
    model_512.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=metrics)
    model_512.summary()
    return model_512
