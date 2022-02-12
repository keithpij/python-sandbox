'''
Main module for RayTrain TensorFlow demo.
Trains an LSTM model locally (on a single process) and distributed using RayTrain.
'''
import argparse
import json
import os
import time

import numpy as np
import ray
from ray.train import Trainer
import tensorflow as tf
from tensorflow import keras

import preprocessor as pre


def get_model_old(config):
    '''
    Creator that returns a LSTM model using Keras.
    Note: Compile is called here.
    '''
    from tensorflow import keras
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


def get_model(config):
    '''
    Creator that returns a LSTM model using Keras.
    Note: Compile is called here.
    '''
    from tensorflow import keras
    vocab_size = config.get('vocab_size')
    embedding_dim = config.get('embedding_dim')
    hidden_dim = config.get('hidden_dim')

    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(input_dim=vocab_size,output_dim=64),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1)
    ])

    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

    return model


def get_data(config):
    '''
    Creator that returns a TensorFlow dataset. This is the best option for both
    local and distributed training.
    '''
    #import tensorflow as tf
    X, y = pre.preprocess_data(config)
    print('Total number of reviews: ', len(X))

    batch_size = config.get('batch_size')

    # Split to create a validation set.
    X_train, y_train, X_valid, y_valid = pre.split_dataset(X, y, 0.8)

    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    valid_dataset = tf.data.Dataset.from_tensor_slices((X_valid, y_valid))

    return train_dataset, valid_dataset


def train_epochs_local(config):
    '''
    Train model locally within a single process.
    '''
    epochs = config.get('epochs')
    batch_size = config.get('batch_size')

    train_dataset, valid_dataset = get_data(config)
    # Repeat is needed to avoid
    train_dataset = train_dataset.batch(batch_size)
    valid_dataset = valid_dataset.batch(batch_size)

    model = get_model(config)

    start_time = time.time()
    history = model.fit(train_dataset, batch_size=batch_size, epochs=epochs)
    duration = time.time() - start_time

    results = history.history
    return model, results, duration


def train_epochs_remote(config):
    per_worker_batch_size = config.get('batch_size')
    epochs = config.get('epochs')
    steps_per_epoch = config.get('steps_per_epoch', 70)

    tf_config = json.loads(os.environ["TF_CONFIG"])
    num_workers = len(tf_config["cluster"]["worker"])
    print(tf_config)

    strategy = tf.distribute.MultiWorkerMirroredStrategy()
    train_dataset, _ = get_data(config)
    global_batch_size = per_worker_batch_size * num_workers
    multi_worker_dataset = train_dataset.batch(global_batch_size)

    with strategy.scope():
        # Model building/compiling need to be within strategy.scope().
        multi_worker_model = get_model(config)

    history = multi_worker_model.fit(
        multi_worker_dataset,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch)

    results = history.history
    return results


def start_ray_train(config, num_workers=4, use_gpu=False):
    '''
    Train model using RayTrain.
    num_workers determines the number of processes.
    Uses the same config as local training.
    '''
    trainer = Trainer(backend="tensorflow", num_workers=num_workers, use_gpu=use_gpu)
    trainer.start()

    start_time = time.time()
    results = trainer.run(train_epochs_remote, config=config)

    print('results:')
    print(results)

    duration = time.time() - start_time

    trainer.shutdown()

    return None, results, duration


def evaluate_model(model, X_test, y_test):
    '''
    Evaluate a model using a test set.
    '''
    score = model.evaluate(X_test, y_test, verbose=0)
    return score


def save_model(model, file_name):
    '''
    Save the model for future use.
    '''
    model.save(file_name)


def load_model(file_name):
    '''
    Load a previously trained model.
    '''
    return keras.models.load_model(file_name, compile=True)


def predict(model, config, pos_or_neg, file_name):
    '''
    Make a prediction based on the model.
    '''
    tokens, text = pre.preprocess_file(config, pos_or_neg, file_name)
    input = np.array(tokens)
    output = model.predict(input)

    print(text)
    print(output[0])


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
        'n_layers': 2,          # TODO: Figure out why this is not used.
        'lr': 0.001,            # TODO: Figure out why this is not used.
        'gpu_available': False
    }

    if args.predict:
        model = load_model(args.model)
        predict(model, config, args.pos_or_neg, args.predict)
        return

    if args.distribute:
        model_state_dict, results, duration = start_ray_train(config, num_workers=4)
        save_model(model_state_dict, 'sa_lstm_distributed')

    if args.local:
        model, results, duration = train_epochs_local(config)
        save_model(model, 'sa_lstm_local.h5')

    # Report results
    print('Smoke Test size: {}'.format(config.get('smoke_test_size')))
    print('Batch size: {}'.format(config.get('batch_size')))
    print('Total elapsed training time: ', duration)
    if args.verbose:
        print(results)


if __name__ == '__main__':
    '''
    Setup all the CLI arguments for this module.
    Some sample commands for calling a model for predictions are below:

        python tensorflow_sa.py -p 0_10.txt -m sa_lstm_local.h5 -pn pos
        python tensorflow_sa.py -p 0_2.txt -m sa_lstm_local.h5 -pn neg
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
