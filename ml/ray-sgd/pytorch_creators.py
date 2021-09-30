import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import preprocessor as pre
from sentiment_lstm import SentimentLSTM


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


def data_creator(config):
    X, y = pre.preprocess_data(config)

    #print('Total number of reviews: ', len(X))
    #analyze_length(X_train)
    #analyze_length(X_valid)

    # Split to create a validation set.
    X_train, y_train, X_valid, y_valid = split(X, y, 0.8)

    # Tensor datasets
    train_data = TensorDataset(torch.from_numpy(np.array(X_train)), torch.from_numpy(np.array(y_train)))
    valid_data = TensorDataset(torch.from_numpy(np.array(X_valid)), torch.from_numpy(np.array(y_valid)))

    # Dataloaders
    batch_size = config.get('batch_size', 50)
    train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
    valid_loader = DataLoader(valid_data, shuffle=True, batch_size=batch_size)
    return train_loader, valid_loader


def model_creator(config):
    # Instantiate the model.
    #print('Model created.')
    vocab_size = config.get('vocab_size')
    output_size = config.get('output_size')
    embedding_dim = config.get('embedding_dim') #400
    hidden_dim = config.get('hidden_dim') #256
    n_layers = config.get('n_layers') #2
    model = SentimentLSTM(vocab_size, output_size, embedding_dim, hidden_dim, n_layers)
    return model


def loss_creator(config):
    return nn.BCELoss()


def optimizer_creator(model, config):
    '''
    Returns optimizer defined upon the model parameters.
    '''
    #return torch.optim.SGD(model.parameters(), lr=config.get("lr", 1e-2))
    return torch.optim.Adam(model.parameters(), lr=config.get("lr", 1e-2))
