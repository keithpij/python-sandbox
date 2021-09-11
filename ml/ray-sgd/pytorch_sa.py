from collections import Counter
import os
from string import punctuation
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from sentiment_lstm import SentimentLSTM


TRAIN_NEGATIVE_REVIEWS_DIR = os.path.join(os.getcwd(), 'aclImdb', 'train', 'neg')
TRAIN_POSITIVE_REVIEWS_DIR = os.path.join(os.getcwd(), 'aclImdb', 'train', 'pos')
NETWORK_FILE_PATH = os.path.join(os.getcwd(), 'lstm.pt')


def get_data(directory, label):
    '''
    Retrieve the Imdb data and load the training set
    '''
    X = []
    y = []
    for file_name in os.listdir(directory):
        file_path = os.path.join(directory, file_name)
        with open(file_path, 'r') as f:
            review = f.read()
        clean = clean_entry(review)
        X.append(clean)
        y.append(label)
    return X, y


def clean_entry(review):
    remove_breaks = review.replace('<br />', ' ')
    lower = remove_breaks.lower()
    #for c in punctuation:
    #    lower = lower.replace(c, ' ')
    valid_characters = [c for c in lower if c not in punctuation]
    cleaned = ''.join(valid_characters)
    return cleaned


def create_tokens(X):
    all_words_string = ' '.join(X)
    words_list = all_words_string.split()
    count_by_word = Counter(words_list)
    total_words = len(count_by_word)
    count_by_word_sorted = count_by_word.most_common(total_words)
    word_to_int_mapping = {w:i+1 for i, (w,c) in enumerate(count_by_word_sorted)}

    return word_to_int_mapping


def tokenize(X, mapping):
    entries_int = []
    for entry_text in X:
        entry_int = [mapping[w] for w in entry_text.split()]
        entries_int.append(entry_int)
    return entries_int


def analyze_length(X):
    lengths = [len(l) for l in X]
    lengths_series = pd.Series(data=lengths)
    print(lengths_series.describe())
    lengths_series.hist()
    plt.show()


def remove_empty(X, y):
    new_X = [X[i] for i, l in enumerate(X) if l>0]
    new_y = [y[i] for i, l in enumerate(X) if l>0]
    X = new_X
    y = new_y


def reshape(X, seq_length):
    ''' 
    Returns a new 2D array of features (X). 
    Each entry is padded with 0's if its number of features are less than seq_lenth.
    If the number of features are greater than seq_lenth then they are truncated to seq_lenth. 
    '''
    features = np.zeros((len(X), seq_length), dtype = int)
    
    for i, review in enumerate(X):
        review_len = len(review)
        
        if review_len <= seq_length:
            zeroes = list(np.zeros(seq_length-review_len))
            new = zeroes + review
        elif review_len > seq_length:
            new = review[0:seq_length]

        features[i,:] = np.array(new)
    
    return features


def split(X, y, train_percent):
    length = len(X)
    X_train = X[0:int(train_percent*length)]
    y_train = y[0:int(train_percent*length)]
    X_valid = X[int(train_percent*length):]
    y_valid = y[int(train_percent*length):]
    return X_train, y_train, X_valid, y_valid


def create_dataloaders(X_train, y_train, X_valid, y_valid, batch_size):
    # Tensor datasets
    train_data = TensorDataset(torch.from_numpy(np.array(X_train)), torch.from_numpy(np.array(y_train)))
    valid_data = TensorDataset(torch.from_numpy(np.array(X_valid)), torch.from_numpy(np.array(y_valid)))

    # Dataloaders
    train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
    valid_loader = DataLoader(valid_data, shuffle=True, batch_size=batch_size)
    return train_loader, valid_loader


def create_network(word_to_int_mapping):
    # Instantiate the model w/ hyperparams
    vocab_size = len(word_to_int_mapping) + 1 # +1 for the 0 padding
    output_size = 1
    embedding_dim = 400
    hidden_dim = 256
    n_layers = 2
    net = SentimentLSTM(vocab_size, output_size, embedding_dim, hidden_dim, n_layers)
    return net


def train_network(net, train_loader, valid_loader, batch_size, train_on_gpu=False):
    # loss and optimization functions
    lr=0.001

    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    # training params
    epochs = 4 # 3-4 is approx where I noticed the validation loss stop decreasing
    counter = 0
    print_every = 100
    clip = 5 # gradient clipping

    # move model to GPU, if available
    if(train_on_gpu):
        net.cuda()

    net.train()
    # train for some number of epochs
    for e in range(epochs):
        # initialize hidden state
        h = net.init_hidden(batch_size)

        # batch loop
        for inputs, labels in train_loader:
            counter += 1

            if(train_on_gpu):
                inputs, labels = inputs.cuda(), labels.cuda()

            # Creating new variables for the hidden state, otherwise
            # we'd backprop through the entire training history
            h = tuple([each.data for each in h])

            # zero accumulated gradients
            net.zero_grad()

            # get the output from the model
            inputs = inputs.type(torch.LongTensor)
            output, h = net(inputs, h)

            # calculate the loss and perform backprop
            loss = criterion(output.squeeze(), labels.float())
            loss.backward()
            # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
            nn.utils.clip_grad_norm_(net.parameters(), clip)
            optimizer.step()

            # loss stats
            if counter % print_every == 0:
                # Get validation loss
                val_h = net.init_hidden(batch_size)
                val_losses = []
                net.eval()
                for inputs, labels in valid_loader:

                    # Creating new variables for the hidden state, otherwise
                    # we'd backprop through the entire training history
                    val_h = tuple([each.data for each in val_h])

                    if(train_on_gpu):
                        inputs, labels = inputs.cuda(), labels.cuda()

                    inputs = inputs.type(torch.LongTensor)
                    output, val_h = net(inputs, val_h)
                    val_loss = criterion(output.squeeze(), labels.float())

                    val_losses.append(val_loss.item())

                net.train()
                print("Epoch: {}/{}...".format(e+1, epochs),
                    "Step: {}...".format(counter),
                    "Loss: {:.6f}...".format(loss.item()),
                    "Val Loss: {:.6f}".format(np.mean(val_losses)))
    return net


def save_network(net):
    torch.save(net, NETWORK_FILE_PATH) 


def load_network():
    net = torch.load(NETWORK_FILE_PATH)
    return net


def get_all_data():
    X_negative, y_negative = get_data(TRAIN_NEGATIVE_REVIEWS_DIR, 0)
    X_positive, y_positive = get_data(TRAIN_POSITIVE_REVIEWS_DIR, 1)
    X = X_negative + X_positive
    y = y_negative + y_positive
    return X, y


def main():
    X, y = get_all_data()
    print('Reviews: ', len(X))

    word_to_int_mapping = create_tokens(X)
    X = tokenize(X, word_to_int_mapping)

    X = reshape(X, 200)
    X_train, y_train, X_valid, y_valid = split(X, y, 0.8)
    #analyze_length(X_train)
    #analyze_length(X_valid)

    # Loaders
    batch_size = 50
    train_loader, valid_loader = create_dataloaders(X_train, y_train, X_valid, y_valid, batch_size)

    # Network
    gpu_available = torch.cuda.is_available()
    net = create_network(word_to_int_mapping)
    print(net)

    # Train
    start_time = time.time()
    trained_net = train_network(net, train_loader, valid_loader, batch_size, gpu_available)
    duration = time.time() - start_time
    save_network(trained_net)
    print('Training time: ', duration)

    #print(word_to_int_mapping)
    #print(X_train[:4])
    #print(X_train[-4:])


if __name__ == "__main__":
    main()