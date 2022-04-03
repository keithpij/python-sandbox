'''
Main module for RayTrain TensorFlow demo.
Trains an LSTM model locally (on a single process) and distributed using RayTrain.
'''
import argparse
import json
import os
import time

import numpy as np
from ray.train import Trainer
import tensorflow as tf
from tensorflow import keras

import preprocessor as pre


def get_model_old(config):
    '''
    Creator that returns a LSTM model using Keras.
    Note: Compile is called here.
    '''
    #from tensorflow import keras
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
    Creates an LSTM model using Keras.
    Note: Compile is called here.
    '''
    from tensorflow import keras
    vocab_size = config['vocab_size']
    embedding_dim = config['embedding_dim']
    hidden_dim = config['hidden_dim']
    output_size = config['output_size']
    lr = config['lr']

    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(input_dim=vocab_size,output_dim=embedding_dim),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(hidden_dim, return_sequences=True)),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(hidden_dim)),
        tf.keras.layers.Dense(hidden_dim, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(output_size)
    ])

    #model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
    model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              optimizer=tf.keras.optimizers.Adam(lr),
              metrics=['accuracy'])

    return model


def get_train_valid_data(config):
    '''
    Returns a TensorFlow dataset after preprocessing the raw training data.
    '''
    batch_size = config['batch_size']

    X_train, y_train, X_valid, y_valid = pre.preprocess_train_valid_data(config)

    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    valid_dataset = tf.data.Dataset.from_tensor_slices((X_valid, y_valid))

    #train_dataset = train_dataset.shuffle(buffer_size).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    #valid_dataset = valid_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    train_dataset = train_dataset.shuffle(1000).batch(batch_size)
    valid_dataset = valid_dataset.shuffle(1000).batch(batch_size)

    return train_dataset, valid_dataset


def get_test_data(config):
    '''
    Returns a TensorFlow dataset after preprocessing the raw test data.
    '''
    batch_size = config.get('batch_size')

    X_test, y_test = pre.preprocess_test_data(config)

    test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))
    test_dataset = test_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    return test_dataset


def train_epochs_local(config):
    '''
    Train model locally within a single process.
    '''
    epochs = config.get('epochs')
    batch_size = config.get('batch_size')

    train_dataset, valid_dataset = get_train_valid_data(config)

    model = get_model(config)

    start_time = time.time()
    history = model.fit(train_dataset, batch_size=batch_size, epochs=epochs, validation_data=valid_dataset, validation_steps=30)
    duration = time.time() - start_time

    results = history.history
    return model, results, duration


def train_epochs_remote(config):
    batch_size = config.get('batch_size')
    epochs = config.get('epochs')

    tf_config = json.loads(os.environ["TF_CONFIG"])
    num_workers = len(tf_config["cluster"]["worker"])
    steps_per_epoch = (batch_size/num_workers)

    # Be sure to call this function before setting up your datasets
    # and your model.
    strategy = tf.distribute.MultiWorkerMirroredStrategy()

    # Get the datasets.
    train_dataset, valid_dataset = get_train_valid_data(config)

    # Get the model. Model building and compiling needs to be done
    # within strategy.scope().
    with strategy.scope():
        model = get_model(config)

    history = model.fit(train_dataset, epochs=epochs,
        steps_per_epoch=steps_per_epoch, validation_data=valid_dataset)

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
    model.save(file_name, save_format='tf')


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

    prediction = output[0]
    sentiment = 'Positive' if prediction > 0.5 else 'Negative'
    print(text)
    print(sentiment)
    print(prediction)


def main(args):
    '''
    Main entry point.
    '''
    # Configuration
    config = {
        'batch_size': 100,          # Batch size for each epoch
        'epochs': 10,               # Total number of epochs
        'embedding_dim': 400,       # Embedded dimension
        'hidden_dim': 256,          # Hidden dimension
        'lr': 0.001,                # Learning Rate
        'output_size': 1,           # Output size
        'sequence_len': 200,        # Number of tokens (words) to put into each review.
        'smoke_test_size': 500,     # Length of training set. 0 for all reviews.
        'vocab_size': 7000          # Vocabulary size
    }

    if args.predict:
        model = load_model(args.model)
        predict(model, config, args.pos_or_neg, args.predict)
        return

    if args.distribute:
        model_state_dict, results, duration = start_ray_train(config, num_workers=4)
        #save_model(model_state_dict, 'sa_lstm_distributed')

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

    Sample commands for training a model locally and distributed:    
        python tensorflow_sa.py -l -v
        python tensorflow_sa.py -d -v

    Sample commands for calling a model for predictions:
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
