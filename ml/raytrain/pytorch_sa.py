'''
Main module for RaySGD PyTorch demo.
Trains an LSTM model locally (on a single process) and distributed using RaySGD and PyTorch.
'''
import argparse
import os
import time

import numpy as np
import ray
import ray.train as train
from ray.train import Trainer
from ray.train.callbacks import JsonLoggerCallback, TBXLoggerCallback
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoad, DistributedSampler, TensorDataset
import torch
import torch.nn as nn

import preprocessor as pre


MODEL_FILE_PATH = os.path.join(os.getcwd(), 'lstm.pt')


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


def train_func(dataloader, model, loss_fn, optimizer, batch_size):
    '''
    This function contains the batch loop. It is used to train
    a model locally and remotely.
    '''
    # initialize hidden state
    h = model.init_hidden(batch_size)
    clip = 5 # used for gradient clipping

    # batch loop
    for inputs, labels in dataloader:

        # Creating new variables for the hidden state, otherwise
        # we'd backprop through the entire training history
        h = tuple([each.data for each in h])

        # zero accumulated gradients
        model.zero_grad()

        # get the output from the model
        inputs = inputs.type(torch.LongTensor)
        output, h = model(inputs, h)

        # calculate the loss and perform backprop
        loss = loss_fn(output.squeeze(), labels.float())
        loss.backward()
        # clip_grad_norm helps prevent the exploding gradient problem in RNNs / LSTMs.
        nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()


def training_setup(config):
    '''
    This function will the datasets, model, loss function, and optimzer.
    '''
    vocab_size = config.get('vocab_size')
    output_size = config.get('output_size')
    embedding_dim = config.get('embedding_dim') #400
    hidden_dim = config.get('hidden_dim') #256
    n_layers = config.get('n_layers') #2
    data_size = config.get("data_size", 1000)
    val_size = config.get("val_size", 400)
    lr = config.get("lr", 1e-2)

    X, y = pre.preprocess_data(config)

    #print('Total number of reviews: ', len(X))
    #analyze_length(X_train)
    #analyze_length(X_valid)

    # Split to create a validation set.
    X_train, y_train, X_valid, y_valid = split(X, y, 0.8)

    # Tensor datasets
    train_dataset = TensorDataset(torch.from_numpy(np.array(X_train)), torch.from_numpy(np.array(y_train)))
    val_dataset = TensorDataset(torch.from_numpy(np.array(X_valid)), torch.from_numpy(np.array(y_valid)))

    model = SentimentLSTM(vocab_size, output_size, embedding_dim, hidden_dim, n_layers)
    
    loss_fn = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    return train_dataset, val_dataset, model, loss_fn, optimizer


def train_local(config):
    '''
    This function trains a model locally. It contains the epoch loop.
    '''
    start_time = time.time()

    train_dataset, val_dataset, model, loss_fn, optimizer = training_setup(config)

    train_loader, _ = cr.data_creator(config)
    batch_size = config.get('batch_size')
    epochs = config.get('epochs')

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size)

    validation_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size)

    # epoch loop
    for _ in range(epochs):
        train_func(train_loader, model, loss_fn, optimizer, batch_size)

    duration = time.time() - start_time
    return model, duration


def train_distributed(config, num_workers=1, use_gpu=False):

    ray.init()

    torch_trainer = TorchTrainer(
        training_operator_cls=SATrainingOperator,
        num_workers=num_workers,
        use_gpu=use_gpu,
        config=config,
        backend="gloo",
        scheduler_step_freq="epoch")

    epochs = config.get('epochs', 4)

    print('Timer started.')
    start_time = time.time()
    for _ in range(epochs):
        stats = torch_trainer.train()
        print(stats)
    duration = time.time() - start_time

    #torch_trainer.validate()

    # If using Ray Client, make sure to force model onto CPU.
    model = torch_trainer.get_model(to_cpu=ray.util.client.ray.is_connected())
    torch_trainer.shutdown()
    return model, duration


def save_model(model):
    torch.save(model, MODEL_FILE_PATH)


def load_model():
    model = torch.load(MODEL_FILE_PATH)
    return model


def main(args):

    # Configuration
    config = {
        'smoke_test_size': 200,  # Length of training set. 0 for all reviews.
        'training_dim': 200,     # Number of tokens (words) to put into each review.
        'vocab_size': 7000,      # Vocabulary size
        'epochs': 4,
        'output_size': 1,
        'embedding_dim': 400,
        'hidden_dim': 256,
        'n_layers': 2,
        'lr': 0.001,
        'batch_size': 10
    }

    # Train
    if args.distribute:
        model, duration = train_distributed(config, num_workers=4)
    else:
        model, duration = train_local(config)

    # Report results
    print('Smoke Test size: {}'.format(config.get('smoke_test_size')))
    print('Total elapsed training time: ', duration)
    print(type(model))
    save_model(model)


if __name__ == "__main__":
    # Setup all the CLI arguments for this module.
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--distribute',
                        help='Distribute the training task.',
                        action='store_true')

    # Parse what was passed in. This will also check the arguments for you and produce
    # a help message if something is wrong.
    args = parser.parse_args()
    main(args)
