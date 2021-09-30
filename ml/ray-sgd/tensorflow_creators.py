'''
This module contains creators used for training the Tensorflow LSTM model.
'''
from tensorflow import keras
import tensorflow as tf
import preprocessor as pre
import numpy as np


def data_creator_np(config):
    '''
    Simple creator that returns a numpy array. This will only work for local training.
    '''
    X, y = pre.preprocess_data(config)
    print('Total number of reviews: ', len(X))
    return np.array(X), np.array(y)

def model_creator(config):
    '''
    Creator that returns a LSTM model using Keras.
    Note: Compile is called here.
    '''

    vocab_size = config.get('vocab_size')
    embedding_dim = config.get('embedding_dim')
    hidden_dim = config.get('hidden_dim')

    model = keras.models.Sequential([
        keras.layers.Embedding(vocab_size, embedding_dim, input_shape=[None]),
        keras.layers.GRU(hidden_dim, return_sequences=True),
        keras.layers.GRU(hidden_dim),
        keras.layers.Dense(1, activation="sigmoid")
    ])

    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

    return model


def data_creator(config):
    '''
    Creator that returns a TensorFlow dataset. This is the best option for both
    local and distributed training.
    '''
    #import tensorflow as tf
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
