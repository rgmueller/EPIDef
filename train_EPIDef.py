import os
import datetime
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
        first layer:  3 convolutional blocks for vertical and horizontal views
        second layer: modified EfficientNet
    """
    model_filter_number = 70
    model_learning_rate = 1e-5
    batch_size = 4
    input_res = 236

    # Define directory for saving checkpoint files:
    directory_ckp = f"epidef_checkpoints\\{network_name}_ckp"
    if not os.path.exists(directory_ckp):
        os.makedirs(directory_ckp)
    if not os.path.exists('epidef_output\\'):
        os.makedirs('epidef_output\\')
    directory_t = f"epidef_output\\{network_name}"
    if not os.path.exists(directory_t):
        os.makedirs(directory_t)
    # txt_name = f"epidef_checkpoints\\lf_{network_name}.txt"

    # Load training data from lightfield .png files:
    print("Loading lightfield paths...")
    # dir_lf_images = ("C:\\Users\\muell\\Google Drive"
    #                  + "\\University\\Master_Project"
    #                  + "\\data_storage\\lightfields")
    dir_lf_images = ("C:\\Users\\muell\\Desktop\\"
                     + "1part_1background")
    list_IDs = get_list_ids(dir_lf_images)

    print("Done loading lightfield paths.")
    fraction = np.int(len(list_IDs)*0.7)
    list_IDs_train, list_IDs_test = list_IDs[:fraction], list_IDs[fraction:]

    model = define_epidef(input_res, input_res, 7, model_filter_number)

    # Write date & time
    # f1 = open(txt_name, 'a')
    now = datetime.datetime.now()

    generator_train = DataGenerator(list_IDs_train,
                                    batch_size=batch_size,
                                    train=False)
    generator_test = DataGenerator(list_IDs_test,
                                   batch_size=batch_size,
                                   train=False)

    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
                                        f"{directory_ckp}\\epidef_model.h5",
                                        save_best_only=True)
    early_stopping_cb = tf.keras.callbacks.EarlyStopping(
                                            patience=10,
                                            restore_best_weights=True)
    callbacks = [checkpoint_cb]
    # , keras.callbacks.TensorBoard(log_dir='./logs')]
    # Try this out at some point:
    # def exponential_decay(lr0, s):
    #     def exponential_decay_fn(epoch):
    #         return lr0 * 0.1**(epoch/s)
    #     return exponential_decay_fn
    # exponential_decay_fn = exponential_decay(0.01, 20)
    # lr_scheduler = tf.keras.callbacks.LearningRateScheduler(
    #                                               exponential_decay_fn)
    model.fit(generator_train,
              epochs=200,
              max_queue_size=10,
              initial_epoch=0,
              verbose=2,
              callbacks=callbacks,
              validation_data=generator_test)

    weight_tmp1 = model.get_weights()
    # model.predict([])
    save_path_file_new = f"{directory_ckp}\\after_last_epoch.hdf5"
    model.save(save_path_file_new)

    print("Weights saved.")
