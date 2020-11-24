import numpy as np

import tensorflow as tf
from epidef_fun.util import get_list_ids
from epidef_fun.epidef_model import define_epidef
from epidef_fun.DataGenerator import DataGenerator

if __name__ == '__main__':

    network_name = 'EPIDef_train'
    iter00 = 0
    load_weights = False
    """
    Model parameters:
        first layer:  3 convolutional blocks
        second layer: 6 convolutional blocks
        last layer:   1 dense block?
    """
    model_filter_number = 70
    model_learning_rate = 1e-5
    batch_size = 1
    input_res = 224

    # Load training data from lightfield .png files:
    print("Loading lightfield paths...")
    dir_lf_images = ("C:\\Users\\muell\\Google Drive\\University\\Master_Project"
                     + "\\data_storage\\lightfields")
    # dir_lf_images = "C:\\Users\\muell\\Desktop\\blender_output_tmp"
    list_IDs = get_list_ids(dir_lf_images)

    print("Done loading lightfield paths.")
    fraction = np.int(len(list_IDs)*0.7)
    list_IDs_train, list_IDs_test = list_IDs[:fraction], list_IDs[fraction:]

    model = define_epidef(input_res, input_res)

    generator_train = DataGenerator(list_IDs_train, batch_size=batch_size, train=True)
    generator_test = DataGenerator(list_IDs_test, batch_size=batch_size, train=False)

    early_stopping_cb = tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
    # callbacks = [checkpoint_cb]  # , keras.callbacks.TensorBoard(log_dir='./logs')]
    # Try this out at some point:
    # def exponential_decay(lr0, s):
    #     def exponential_decay_fn(epoch):
    #         return lr0 * 0.1**(epoch/s)
    #     return exponential_decay_fn
    # exponential_decay_fn = exponential_decay(0.01, 20)
    # lr_scheduler = tf.keras.callbacks.LearningRateScheduler(exponential_decay_fn)
    model.fit(generator_train,
              epochs=100,
              max_queue_size=10,
              initial_epoch=0,
              verbose=2,
              # callbacks=callbacks,
              validation_data=generator_test)
