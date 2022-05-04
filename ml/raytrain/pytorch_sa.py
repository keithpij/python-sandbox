'''
Main module for RaySGD PyTorch demo.
Trains an LSTM model locally (on a single process) and distributed using RaySGD and PyTorch.
'''
import argparse
import os
import threading
import time
from matplotlib.pyplot import bar_label

import numpy as np
import ray
import ray.train as train
from ray.train import Trainer
from ray.train.callbacks import JsonLoggerCallback, TBXLoggerCallback
from sklearn.utils import shuffle
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler, TensorDataset
import torch
import torch.nn as nn

import preprocessor as pre


class SentimentLSTM(nn.Module):
    '''
    An LSTM is a type of RNN network that can be used to perform Sentiment analysis.
    '''

    def __init__(self, vocab_size, output_dim, embedding_dim, hidden_dim, n_layers, batch_size, dropout_prob):
        '''
        Initialize the model and set up the layers.
        '''
        super(SentimentLSTM, self).__init__()

        self.output_dim = output_dim
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.vocab_size = vocab_size
        self.dropout_prob = dropout_prob

        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # LSTM Layer
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim, num_layers=n_layers, batch_first=True)

        #self.hidden = self.init_hidden()
        self.dropout = nn.Dropout(dropout_prob)

        # Linear layer
        self.fcl = nn.Linear(hidden_dim, output_dim)

        # Sigmoid layer
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

        # fully-connected layer
        out = self.dropout(lstm_out)
        out = self.fcl(out)

        # sigmoid function
        sig_out = self.sig(out)

        # reshape to be batch_size first
        sig_out = sig_out.view(batch_size, -1)
        sig_out = sig_out[:, -1] # get last batch of labels

        # return last sigmoid output and hidden state
        return sig_out, hidden

    def init_hidden(self, batch_size=None):
        ''' 
        Initializes hidden state
        Creates two new tensors with sizes n_layers x batch_size x hidden_dim,
        initialized to zero, for hidden state and cell state of LSTM.

        Note: The batch_size needs to be 1 for predictions.
        '''
        if not batch_size:
            batch_size = self.batch_size

        h0 = torch.zeros((self.n_layers, batch_size, self.hidden_dim))
        c0 = torch.zeros((self.n_layers, batch_size, self.hidden_dim))
        hidden = (h0,c0)
        return hidden

def training_setup(config):
    '''
    This function will the datasets, model, loss function, and optimzer.
    '''
    vocab_size = config['vocab_size']
    output_dim = config['output_dim']
    embedding_dim = config['embedding_dim']
    hidden_dim = config['hidden_dim']
    n_layers = config['n_layers']
    lr = config['lr']
    batch_size = config['batch_size']

    X_train, y_train, X_valid, y_valid = pre.preprocess_train_valid_data(config)

    # Tensor datasets
    train_dataset = TensorDataset(torch.from_numpy(np.array(X_train)), torch.from_numpy(np.array(y_train)))
    val_dataset = TensorDataset(torch.from_numpy(np.array(X_valid)), torch.from_numpy(np.array(y_valid)))

    model = SentimentLSTM(vocab_size, output_dim, embedding_dim, hidden_dim, n_layers, batch_size)
    
    loss_fn = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    return train_dataset, val_dataset, model, loss_fn, optimizer


def train_batches(dataloader, model, loss_fn, optimizer, config):
    '''
    This function contains the batch loop. It is used to train
    a model locally and remotely.
    '''
    grad_clip = config['grad_clip']

    # Initialize the hidden state.
    h = model.init_hidden()

    # batch loop
    for inputs, labels in dataloader:
        # Create new variables for the hidden state, otherwise we
        # would backprop through the entire training history.
        h = tuple([each.data for each in h])
        # zero accumulated gradients
        model.zero_grad()

        # get the output from the model
        #inputs = inputs.type(torch.LongTensor)
        output, h = model(inputs, h)

        # calculate the loss and perform back propagation
        loss = loss_fn(output.squeeze(), labels.float())
        loss.backward()
        
        # clip_grad_norm helps prevent the exploding gradient problem in RNNs / LSTMs.
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()


def validate_epoch(dataloader, model, loss_fn):
    # Initialize the hidden state.
    h = model.init_hidden()

    num_batches = len(dataloader)
    model.eval()
    loss = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            h = tuple([each.data for each in h])
            # zero accumulated gradients
            model.zero_grad()

            # get the output from the model
            #inputs = inputs.type(torch.LongTensor)
            output, h = model(inputs, h)

            # calculate the loss and perform backprop
            loss += loss_fn(output.squeeze(), labels.float())

            #pred = model(inputs)
            #loss += loss_fn(pred, labels).item()

    loss /= num_batches
    result = {'process_id': os.getpid(), 'loss': loss}
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
        shuffle = True,
        batch_size=batch_size)

    validation_loader = torch.utils.data.DataLoader(
        val_dataset,
        shuffle = True,
        batch_size=batch_size)

    # epoch loop
    results = []
    for epoch in range(epochs):
        train_batches(train_loader, model, loss_fn, optimizer, config)
        result = validate_epoch(validation_loader, model, loss_fn)
        result['epoch'] = epoch + 1
        results.append(result)

    duration = time.time() - start_time
    return model.state_dict(), results, duration


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

    # epoch loop
    results = []
    for epoch in range(epochs):
        train_batches(train_loader, model, loss_fn, optimizer, config)
        result = validate_epoch(validation_loader, model, loss_fn)
        result['epoch'] = epoch + 1
        train.report(**result)
        results.append(result)

    return model.state_dict(), results


def start_ray_train(config, num_workers=4, use_gpu=False):
    '''
    This function starts Ray Train. 
    num_workers determines the number of processes.
    Uses the same config as local training.
    '''
    trainer = Trainer(backend="torch", num_workers=num_workers, use_gpu=use_gpu)
    trainer.start()

    start_time = time.time()
    response = trainer.run(train_epochs_remote, config)
    duration = time.time() - start_time

    trainer.shutdown()

    model_state_dict = response[0][0]
    results = []
    for (_, result) in response:
        results.append(result)

    return model_state_dict, results, duration


def save_model(model_state_dict, file_name):
    file_path = os.path.join(os.getcwd(), file_name)
    torch.save(model_state_dict, file_path)


def load_model(file_name, config):
    file_path = os.path.join(os.getcwd(), file_name)

    vocab_size = config.get('vocab_size')
    output_dim = config.get('output_dim')
    embedding_dim = config.get('embedding_dim') #400
    hidden_dim = config.get('hidden_dim') #256
    n_layers = config.get('n_layers') #2
    batch_size = config.get('batch_size')

    model = SentimentLSTM(vocab_size, output_dim, embedding_dim, hidden_dim, n_layers, batch_size)
    model.load_state_dict(torch.load(file_path))
    return model


def predict(model, config, pos_or_neg, file_name):
    '''
    Make a prediction based on the model.
    '''
    tokens, text = pre.preprocess_file(config, pos_or_neg, file_name)
    tensor_input = torch.from_numpy(np.array(tokens))
    h = model.init_hidden(batch_size=1)
    h = tuple([each.data for each in h])
    output, h = model(tensor_input, h)

    prediction = output.item()
    sentiment = 'Positive' if prediction > 0.5 else 'Negative'
    print(text)
    print(sentiment)
    print(output.item())


def main(args):
    '''
    Main entry point.
    '''

    # Configuration
    config = {
        'batch_size': 100,          # Batch size for each epoch
        'dropout_prob': .5,         # Dropout probability
        'embedding_dim': 400,       # Embedded dimension
        'epochs': 4,                # Total number of epochs
        'grad_clip': 5,             # Gradient Clip
        'hidden_dim': 256,          # Hidden dimension
        'lr': 0.001,                # Learning Rate
        'n_layers': 2,              # Number of hidden layers in the LSTM
        'output_dim': 1,            # Output dimension
        'sequence_len': 200,        # Number of tokens (words) to put into each review.
        'smoke_test_size': 500,     # Length of training set. 0 for all reviews.
        'vocab_size': 7000          # Vocabulary size
    }

    if args.predict:
        model = load_model(args.model, config)
        predict(model, config, args.pos_or_neg, args.predict)
        return

    if args.distribute:
        model_state_dict, results, duration = start_ray_train(config, num_workers=4)
        save_model(model_state_dict, 'sa_lstm_distributed.pt')

    if args.local:
        model_state_dict, results, duration = train_epochs_local(config)
        save_model(model_state_dict, 'sa_lstm_local.pt')

    # Report results
    print('Smoke Test size: {}'.format(config.get('smoke_test_size')))
    print('Batch size: {}'.format(config.get('batch_size')))
    print('Total elapsed training time: ', duration)
    if args.verbose:
        print(results)


if __name__ == "__main__":
    '''
    Setup all the CLI arguments for this module.

    Sample commands for training a model locally and distributed:    
        python pytorch_sa.py -l -v
        python pytorch_sa.py -d -v

    Sample commands for using a trained model to make predictions:
        python pytorch_sa.py -p 0_10.txt -m sa_lstm_local.pt -pn pos
        python pytorch_sa.py -p 0_2.txt -m sa_lstm_local.pt -pn neg
    '''

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--distribute',
                        help='Train model using distributed workers.',
                        action='store_true')
    parser.add_argument('-l', '--local',
                        help='Train model locally.',
                        action='store_true')
    parser.add_argument('-m', '--model',
                        help='Pre-trained model to load.')
    parser.add_argument('-p', '--predict',
                        help='Make a prediction using a pre-trained model. Specify a file from the test set.')
    parser.add_argument('-pn', '--pos_or_neg',
                        help='Positive or negative. Specify where the test set file is located.')
    parser.add_argument('-v', '--verbose',
                        help='Verbose output (show results list).',
                        action='store_true')

    # Parse what was passed in. This will also check the arguments for you and produce
    # a help message if something is wrong.
    args = parser.parse_args()
    main(args)
