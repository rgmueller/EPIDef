import numpy as np
import tensorflow.keras as keras
from epidef_fun.util import load_lightfield_data
from epidef_fun.generate_traindata import generate_traindata, data_augmentation


class DataGenerator(keras.utils.Sequence):
    """Generates data for Keras"""
    def __init__(self, list_ids, batch_size=1, dim=(400, 400, 7),
                 n_classes=3, shuffle=True, train=True):
        """Initialization"""
        self.dim = dim
        self.batch_size = batch_size
        self.list_IDs = list_ids
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.train = train
        self.on_epoch_end()

    def __len__(self):
        """Denotes the number of batches per epoch"""
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        """Generate one batch of data"""
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_ids_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        x, y = self.__data_generation(list_ids_temp)

        return x, y

    def on_epoch_end(self):
        """Updates indexes after each epoch"""
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_ids_temp):
        """Generates data containing batch_size samples"""
        # X : (n_samples, *dim, n_channels)
        # Generate data
        x, y = load_lightfield_data(list_ids_temp, 224)
        (x_vert, x_hori, y) = generate_traindata(x, y, 224, self.batch_size, 7)

        (x_vert, x_hori, y) = data_augmentation(x_vert, x_hori, y,
                                                self.batch_size, self.train)
        y_cat = keras.utils.to_categorical(y, num_classes=self.n_classes)
        # print(y_cat)
        return ([x_vert, x_hori], y_cat)
