from collections import Counter
import os
from string import punctuation

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


TRAIN_NEGATIVE_REVIEWS_DIR = os.path.join(os.getcwd(), 'aclImdb', 'train', 'neg')
TRAIN_POSITIVE_REVIEWS_DIR = os.path.join(os.getcwd(), 'aclImdb', 'train', 'pos')


def get_all_data():
    '''
    Load all the raw negative and positive data from the review files.
    '''
    X_negative, y_negative = get_data(TRAIN_NEGATIVE_REVIEWS_DIR, 0)
    X_positive, y_positive = get_data(TRAIN_POSITIVE_REVIEWS_DIR, 1)
    X = X_negative + X_positive
    y = y_negative + y_positive
    return X, y


def get_data(directory, label):
    '''
    Retrieve the Imdb data from the specified folder.
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


def create_tokens(X, vocab_size):
    all_words_string = ' '.join(X)
    words_list = all_words_string.split()
    count_by_word = Counter(words_list)
    total_words = len(count_by_word)
    count_by_word_sorted = count_by_word.most_common(total_words)
    word_to_int_mapping = {}
    for i, (w,c) in enumerate(count_by_word_sorted):
        if i == vocab_size-1:  # Leave one for 0 padding.
            break
        word_to_int_mapping[w] = i + 1

    #word_to_int_mapping = {w:i+1 for i, (w,c) in enumerate(count_by_word_sorted)}

    #print(word_to_int_mapping)
    #bra
    return word_to_int_mapping


def tokenize(X, mapping):
    entries_int = []
    for entry_text in X:
        entry_int = []
        for w in entry_text.split():
            if w in mapping:
                entry_int.append(mapping[w])
        #entry_int = [mapping[w] for w in entry_text.split()]
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


def preprocess_data(config):
    X, y = get_all_data()

    smoke_test_size = config.get('smoke_test_size', 0)
    vocab_size = config.get('vocab_size')

    if smoke_test_size:
        X = X[:smoke_test_size]
        y = y[:smoke_test_size]

    word_to_int_mapping = create_tokens(X, vocab_size)
    print(len(word_to_int_mapping))

    X = tokenize(X, word_to_int_mapping)

    # Set vocab_size into config so that it can be used by the network.
    #config['vocab_size'] = len(word_to_int_mapping) + 1 # Add one to account for 0 padding.

    training_dim = config.get('training_dim', 200)
    X = reshape(X, training_dim)
    return X, y


def split_dataset(X, y, train_percent):
    '''
    Splits X, and y producing a training set that is the specified percentage.
    The remaining rows will become a part of the validation set.
    '''
    length = len(X)
    X_train = X[0:int(train_percent*length)]
    y_train = y[0:int(train_percent*length)]
    X_valid = X[int(train_percent*length):]
    y_valid = y[int(train_percent*length):]
    return X_train, y_train, X_valid, y_valid
