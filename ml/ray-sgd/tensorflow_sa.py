'''
RaySGD demo for TensorFlow.
'''
import argparse
import os
import time

import numpy as np
import ray
from ray.util.sgd.tf.tf_trainer import TFTrainer
from tensorflow import keras

import tensorflow_creators as cr


MODEL_FOLDER = os.path.join(os.getcwd(), 'keras')


def train_local(config):
    epochs = config.get('epochs')
    batch_size = config.get('batch_size')

    #X_train, y_train = cr.data_creator(config)
    train_dataset, val_dataset = cr.data_creator(config)
    model = cr.model_creator(config)

    start_time = time.time()
    #history = model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.2)
    #history = model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs)
    history = model.fit(train_dataset, batch_size=batch_size, epochs=epochs)
    print(history)
    duration = time.time() - start_time

    return model, duration


def train_distributed(config, num_replicas=4, use_gpu=False):

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
    for i in range(epochs):
        stats = trainer.train()
        #print(stats)
    duration = time.time() - start_time

    model = trainer.get_model()
    trainer.shutdown()
    return model, duration


def evaluate_model(model, x_test, y_test):
    score = model.evaluate(x_test, y_test, verbose=0)
    return score


def save_model(model):
    keras.models.save_model(model, MODEL_FOLDER)


def load_model():
    return keras.models.load_model(MODEL_FOLDER, compile=True)


def predict(model, image_samples):
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

    # Configuration
    config = {
        'smoke_test_size': 200,  # Length of training set. 0 for all reviews.
        'training_dim': 200,     # Number of tokens (words) to put into each review.
        'vocab_size': 7000,      # Vocabulary size
        'epochs': 4,
        'output_size': 1,
        'embedding_dim': 400,
        'hidden_dim': 256,
        'batch_size': 10
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
