'''
Main module for RaySGD TensorFlow demo.
Trains an LSTM model locally (on a single process) and distributed using RaySGD.
'''
import argparse
import os
import time

import ray
from ray.util.sgd.tf.tf_trainer import TFTrainer
from tensorflow import keras
import tensorflow_creators as cr
import numpy as np


MODEL_FOLDER = os.path.join(os.getcwd(), 'keras')


def train_local(config):
    '''
    Train model locally within a single process.
    '''
    epochs = config.get('epochs')
    batch_size = config.get('batch_size')

    train_dataset, _ = cr.data_creator(config)
    model = cr.model_creator(config)

    start_time = time.time()
    history = model.fit(train_dataset, batch_size=batch_size, epochs=epochs)
    #print(history)
    duration = time.time() - start_time

    return model, duration


def train_distributed(config, num_replicas=4, use_gpu=False):
    '''
    Train model using RaySGD.
    num_replicas determines the number of processes.
    Uses the same config as local training.
    '''
    ray.init()

    trainer = TFTrainer(
        model_creator=cr.model_creator,
        data_creator=cr.data_creator,
        num_replicas=num_replicas,
        use_gpu=use_gpu,
        verbose=True,
        config=config)

    epochs = config.get('epochs', 4)

    print('Timer started.')
    start_time = time.time()
    for _ in range(epochs):
        stats = trainer.train()
        #print(stats)
    duration = time.time() - start_time

    model = trainer.get_model()
    trainer.shutdown()
    return model, duration


def evaluate_model(model, X_test, y_test):
    '''
    Evaluate a model using a test set.
    '''
    score = model.evaluate(X_test, y_test, verbose=0)
    return score


def save_model(model):
    '''
    Save the model for future use.
    '''
    keras.models.save_model(model, MODEL_FOLDER)


def load_model():
    '''
    Load a previously trained model.
    '''
    return keras.models.load_model(MODEL_FOLDER, compile=True)


def predict(model, image_samples):
    '''
    Make predictions using a previously trained model.
    '''
    # A few random samples
    samples_to_predict = []

    # Convert into Numpy array
    samples_to_predict = np.array(image_samples)
    print(samples_to_predict.shape)

    probabilities = model.predict(samples_to_predict)
    print(type(probabilities))
    print(probabilities)

    # Generate arg maxes for predictions
    classes = np.argmax(probabilities, axis=1)
    print(classes)


def main(args):
    '''
    Main entry point.
    '''
    # Configuration
    config = {
        'smoke_test_size': 200,  # Length of training set. 0 for all reviews.
        'training_dim': 200,     # Number of tokens (words) to put into each review.
        'vocab_size': 7000,      # Vocabulary size
        'epochs': 4,
        'output_size': 1,
        'embedding_dim': 400,
        'hidden_dim': 256,
        'batch_size': 10,
        'gpu_available': False
    }

    # Train
    if args.distribute:
        model, duration = train_distributed(config, num_replicas=4)
    else:
        model, duration = train_local(config)

    # Report results
    print('Smoke Test size: {}'.format(config.get('smoke_test_size')))
    print('Batch size: {}'.format(config.get('batch_size')))
    print('Total elapsed training time: ', duration)
    print(type(model))
    #save_model(model)


if __name__ == '__main__':
    # Setup all the CLI arguments for this module.
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--distribute',
                        help='Distribute the training task.',
                        action='store_true')

    # Parse what was passed in. This will also check the arguments for you and produce
    # a help message if something is wrong.
    args = parser.parse_args()
    main(args)
