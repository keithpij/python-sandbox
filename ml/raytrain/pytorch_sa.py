'''
Main module for RaySGD PyTorch demo.
Trains an LSTM model locally (on a single process) and distributed using RaySGD and PyTorch.
'''
import argparse
import os
import threading
import time

import numpy as np
import ray
import ray.train as train
from ray.train import Trainer
from ray.train.callbacks import JsonLoggerCallback, TBXLoggerCallback
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler, TensorDataset
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


def init_hidden(model, config, train_on_gpu=False):
    ''' Initializes hidden state '''
    # Create two new tensors with sizes n_layers x batch_size x hidden_dim,
    # initialized to zero, for hidden state and cell state of LSTM
    hidden_dim = config.get('hidden_dim') #256
    n_layers = config.get('n_layers') #2
    batch_size = config.get('batch_size')

    weight = next(model.parameters()).data

    if train_on_gpu:
        hidden = (weight.new(n_layers, batch_size, hidden_dim).zero_().cuda(),
                weight.new(n_layers, batch_size, hidden_dim).zero_().cuda())
    else:
        hidden = (weight.new(n_layers, batch_size, hidden_dim).zero_(),
                    weight.new(n_layers, batch_size, hidden_dim).zero_())

    return hidden


def training_setup(config):
    '''
    This function will the datasets, model, loss function, and optimzer.
    '''
    vocab_size = config.get('vocab_size')
    output_size = config.get('output_size')
    embedding_dim = config.get('embedding_dim') #400
    hidden_dim = config.get('hidden_dim') #256
    n_layers = config.get('n_layers') #2
    lr = config.get("lr", 1e-2)

    X, y = pre.preprocess_data(config)

    #print('Total number of reviews: ', len(X))
    #analyze_length(X_train)
    #analyze_length(X_valid)

    # Split to create a validation set.
    X_train, y_train, X_valid, y_valid = pre.split_dataset(X, y, 0.8)

    # Tensor datasets
    train_dataset = TensorDataset(torch.from_numpy(np.array(X_train)), torch.from_numpy(np.array(y_train)))
    val_dataset = TensorDataset(torch.from_numpy(np.array(X_valid)), torch.from_numpy(np.array(y_valid)))

    model = SentimentLSTM(vocab_size, output_size, embedding_dim, hidden_dim, n_layers)
    
    loss_fn = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    return train_dataset, val_dataset, model, loss_fn, optimizer


def train_batches(dataloader, model, loss_fn, optimizer, config):
    '''
    This function contains the batch loop. It is used to train
    a model locally and remotely.
    '''
    # initialize hidden state
    h = init_hidden(model, config)
    clip = config.get('grad_clip') # used for gradient clipping

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


def validate_epoch(dataloader, model, loss_fn):
    num_batches = len(dataloader)
    model.eval()
    loss = 0
    with torch.no_grad():
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
            loss += loss_fn(output.squeeze(), labels.float())

            #pred = model(inputs)
            #loss += loss_fn(pred, labels).item()

    loss /= num_batches
    result = {'thread_id': threading.current_thread().name, 'loss': loss}
    return result


def train_epochs_local(config):
    '''
    This function trains a model locally. It contains the epoch loop.
    '''
    start_time = time.time()

    train_dataset, val_dataset, model, loss_fn, optimizer = training_setup(config)

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
        train_batches(train_loader, model, loss_fn, optimizer, config)

    duration = time.time() - start_time
    return model, duration


def train_epochs_remote(config):
    '''
    This function will be run on each remote worker. It contains the epoch loop.
    '''
    train_dataset, val_dataset, model, loss_fn, optimizer = training_setup(config)

    batch_size = config.get('batch_size')
    epochs = config.get('epochs')

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=DistributedSampler(train_dataset))

    validation_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        sampler=DistributedSampler(val_dataset))

    # Prepare the data and the model for distributed training.
    train_loader = train.torch.prepare_data_loader(train_loader)
    validation_loader = train.torch.prepare_data_loader(validation_loader)
    model = train.torch.prepare_model(model)
    #model = DistributedDataParallel(model)

    # epoch loop
    results = []
    for _ in range(epochs):
        train_batches(train_loader, model, loss_fn, optimizer, config)
        result = validate_epoch(validation_loader, model, loss_fn)
        train.report(**result)
        results.append(result)

    return results


def start_ray_train(config, num_workers=4, use_gpu=False):
    '''
    This function starts Ray Train. 
    num_workers determines the number of processes.
    Uses the same config as local training.
    '''
    trainer = Trainer(backend="torch", num_workers=num_workers, use_gpu=use_gpu)
    trainer.start()

    start_time = time.time()
    results = trainer.run(train_epochs_remote, config)

    print('results:')
    print(results)

    duration = time.time() - start_time

    trainer.shutdown()

    return None, duration


def save_model(model):
    torch.save(model, MODEL_FILE_PATH)


def load_model():
    model = torch.load(MODEL_FILE_PATH)
    return model


def main(args):
    '''
    Main entry point.
    '''
    # Configuration
    config = {
        'smoke_test_size': 200,  # Length of training set. 0 for all reviews.
        'epochs': 4,             # Total number of epochs
        'batch_size': 10,        # Batch size for each epoch
        'training_dim': 200,     # Number of tokens (words) to put into each review.
        'vocab_size': 7000,      # Vocabulary size
        'output_size': 1,
        'embedding_dim': 400,
        'hidden_dim': 256,
        'n_layers': 2,
        'lr': 0.001,
        'grad_clip': 5
    }

    # Start Training
    if args.distribute:
        model, duration = start_ray_train(config, num_workers=4)
    else:
        model, duration = train_epochs_local(config)

    # Report results
    print('Smoke Test size: {}'.format(config.get('smoke_test_size')))
    print('Batch size: {}'.format(config.get('batch_size')))
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
