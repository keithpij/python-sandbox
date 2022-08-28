from collections import Counter
import json
import os
from string import punctuation
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from torch import seed

DATASETS_DIR = os.path.join(os.pardir, os.pardir, 'datasets')
TRAIN_NEGATIVE_REVIEWS_DIR = os.path.join(DATASETS_DIR, 'aclImdb', 'train', 'neg')
TRAIN_POSITIVE_REVIEWS_DIR = os.path.join(DATASETS_DIR, 'aclImdb', 'train', 'pos')
TEST_NEGATIVE_REVIEWS_DIR = os.path.join(DATASETS_DIR, 'aclImdb', 'test', 'neg')
TEST_POSITIVE_REVIEWS_DIR = os.path.join(DATASETS_DIR, 'aclImdb', 'test', 'pos')

TEST_REVIEWS_DIR = os.path.join(os.getcwd(), 'aclImdb', 'test')


def get_train_valid_data(smoke_test_size=0):
    '''
    Load all the raw negative and positive data from the review files.
    If data is needed for a quick experiment (smoke test) then we want to get an equal amount of files
    from the negative dir and the positive dir.
    '''
    max_files = 0
    if smoke_test_size:
        max_files = smoke_test_size/2

    X_negative, y_negative = read_files(TRAIN_NEGATIVE_REVIEWS_DIR, 0, max_files)
    X_positive, y_positive = read_files(TRAIN_POSITIVE_REVIEWS_DIR, 1, max_files)
    X = X_negative + X_positive
    y = y_negative + y_positive

    X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, shuffle=True, stratify=y, random_state=42)

    return X_train, y_train, X_valid, y_valid


def get_test_data():
    '''
    Load all the raw negative and positive test data from the review files.
    '''
    X_negative, y_negative = read_files(TEST_NEGATIVE_REVIEWS_DIR, 0)
    X_positive, y_positive = read_files(TEST_POSITIVE_REVIEWS_DIR, 1)
    X_test = X_negative + X_positive
    y_test = y_negative + y_positive

    return X_test, y_test


def read_files(directory, label, max_files=0):
    '''
    Retrieve the Imdb data from the specified folder.
    '''
    count = 0
    X = []
    y = []
    for file_name in os.listdir(directory):
        file_path = os.path.join(directory, file_name)
        with open(file_path, 'r') as f:
            review = f.read()
        clean = clean_entry(review)
        X.append(clean)
        y.append(label)
        count += 1
        if max_files and count >= max_files:
            break

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
        entry_int = tokenize_text(entry_text, mapping)
        #entry_int = []
        #for w in entry_text.split():
        #    if w in mapping:
        #        entry_int.append(mapping[w])
        #entry_int = [mapping[w] for w in entry_text.split()]
        entries_int.append(entry_int)
    return entries_int


def tokenize_text(text, mapping):
    entry_int = []
    for w in text.split():
        if w in mapping:
            entry_int.append(mapping[w])
    return entry_int


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
    features = np.zeros((len(X), seq_length), dtype=int)

    for i, review in enumerate(X):
        review_len = len(review)

        if review_len <= seq_length:
            zeroes = list(np.zeros(seq_length-review_len, dtype=int))
            new = review + zeroes
        elif review_len > seq_length:
            new = review[0:seq_length]

        features[i,:] = np.array(new)

    return features


def preprocess_train_valid_data(config):
    smoke_test_size = config['smoke_test_size']
    vocab_size = config['vocab_size']

    X_train, y_train, X_valid, y_valid = get_train_valid_data(smoke_test_size)

    word_to_int_mapping = create_tokens(X_train, vocab_size)
 
    with open('word_to_int_mapping.json', 'w') as json_file:
        json.dump(word_to_int_mapping, json_file)

    X_train = tokenize(X_train, word_to_int_mapping)
    X_valid = tokenize(X_valid, word_to_int_mapping)

    sequence_len = config['sequence_len']
    X_train = reshape(X_train, sequence_len)
    X_valid = reshape(X_valid, sequence_len)

    return X_train, y_train, X_valid, y_valid


def preprocess_test_data(config):
    smoke_test_size = config['smoke_test_size']
    vocab_size = config['vocab_size']

    X_test, y_test = get_test_data(smoke_test_size)

    with open('word_to_int_mapping.json') as json_file:
        word_to_int_mapping = json.load(json_file)
    assert len(word_to_int_mapping) == vocab_size-1

    X_test = tokenize(X_test, word_to_int_mapping)

    sequence_len = config.get('sequence_len', 200)
    X_test = reshape(X_test, sequence_len)
    return X_test, y_test


def preprocess_text(config, text):
    '''
    Preprocess raw review text entered by a user.
    '''
    vocab_size = config.get('vocab_size')
    sequence_len = config.get('sequence_len', 200)

    with open('word_to_int_mapping.json') as json_file:
        word_to_int_mapping = json.load(json_file)
    assert len(word_to_int_mapping) == vocab_size-1

    text = clean_entry(text)
    tokens = tokenize_text(text, word_to_int_mapping)

    num_tokens = len(tokens)

    if num_tokens <= sequence_len:
        zeroes = list(np.zeros(sequence_len-num_tokens, dtype=int))
        new = tokens + zeroes
    elif num_tokens > sequence_len:
        new = tokens[0:sequence_len]
    tokens = new    

    return [tokens]


def preprocess_file(config, pos_or_neg, file_name):
    '''
    Preprocess raw review text found in a single file found within the test set.
    '''
    vocab_size = config.get('vocab_size')
    sequence_len = config.get('sequence_len', 200)

    with open('word_to_int_mapping.json') as json_file:
        word_to_int_mapping = json.load(json_file)
    assert len(word_to_int_mapping) == vocab_size-1

    # Load a single file from the test set.
    file_path = os.path.join(TEST_REVIEWS_DIR, pos_or_neg, file_name)
    with open(file_path, 'r') as f:
        review = f.read()
    text = clean_entry(review)

    tokens = tokenize_text(text, word_to_int_mapping)

    num_tokens = len(tokens)

    if num_tokens <= sequence_len:
        zeroes = list(np.zeros(sequence_len-num_tokens, dtype=int))
        new = tokens + zeroes
    elif num_tokens > sequence_len:
        new = tokens[0:sequence_len]
    tokens = new    

    return [tokens], review


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
