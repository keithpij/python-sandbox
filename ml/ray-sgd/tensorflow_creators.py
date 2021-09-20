import numpy as np

import preprocessor as pre
import pytorch_creators as cr

def data_creator_np(config):
    X, y = pre.preprocess_data(config)
    print('Total number of reviews: ', len(X))
    return np.array(X), np.array(y)


def data_creator2(config):
    import tensorflow as tf
    X, y = pre.preprocess_data(config)
    print('Total number of reviews: ', len(X))

    batch_size = config.get('batch_size')

    # Split to create a validation set.
    X_train, y_train, X_valid, y_valid = split(X, y, 0.8)

    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    valid_dataset = tf.data.Dataset.from_tensor_slices((X_valid, y_valid))

    # Repeat is needed to avoid
    train_dataset = train_dataset.batch(batch_size)
    valid_dataset = valid_dataset.batch(batch_size)
    return train_dataset, valid_dataset


def split(X, y, train_percent):
    '''
    Splits X, and y producing a training set that is the specified percentage.
    The remaining row will become a part of the validation set.
    '''
    length = len(X)
    X_train = X[0:int(train_percent*length)]
    y_train = y[0:int(train_percent*length)]
    X_valid = X[int(train_percent*length):]
    y_valid = y[int(train_percent*length):]
    return X_train, y_train, X_valid, y_valid

