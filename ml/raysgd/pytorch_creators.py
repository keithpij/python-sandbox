import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import preprocessor as pre


class SentimentLSTM(nn.Module):
    '''
    An LSTM is a type of RNN network the  that will be used to perform Sentiment analysis.
    '''

    def __init__(self, vocab_size, output_size, embedding_dim, hidden_dim, n_layers, drop_prob=0.5):
        '''
        Initialize the model and set up the layers.
        '''
        super().__init__()

        self.output_size = output_size
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim

        # embedding and LSTM layers
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers,
                            dropout=drop_prob, batch_first=True)

        # dropout layer
        self.dropout = nn.Dropout(0.3)

        # linear and sigmoid layers
        self.fcl = nn.Linear(hidden_dim, output_size)
        self.sig = nn.Sigmoid()

    def forward(self, x, hidden):
        '''
        Forward pass
        '''
        batch_size = x.size(0)

        # embeddings and lstm_out
        embeds = self.embedding(x)
        lstm_out, hidden = self.lstm(embeds, hidden)

        # stack up lstm outputs
        lstm_out = lstm_out.contiguous().view(-1, self.hidden_dim)

        # dropout and fully-connected layer
        out = self.dropout(lstm_out)
        out = self.fcl(out)
        # sigmoid function
        sig_out = self.sig(out)

        # reshape to be batch_size first
        sig_out = sig_out.view(batch_size, -1)
        sig_out = sig_out[:, -1] # get last batch of labels

        # return last sigmoid output and hidden state
        return sig_out, hidden


    def init_hidden(self, batch_size, train_on_gpu=False):
        ''' Initializes hidden state '''
        # Create two new tensors with sizes n_layers x batch_size x hidden_dim,
        # initialized to zero, for hidden state and cell state of LSTM
        weight = next(self.parameters()).data

        if train_on_gpu:
            hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().cuda(),
                  weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().cuda())
        else:
            hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_(),
                      weight.new(self.n_layers, batch_size, self.hidden_dim).zero_())

        return hidden


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
