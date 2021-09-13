import argparse
import os
import time

import numpy as np
import ray
from ray.util.sgd import TorchTrainer
import torch
import torch.nn as nn

from sa_training_operator import SATrainingOperator
import pytorch_creators as cr


NETWORK_FILE_PATH = os.path.join(os.getcwd(), 'lstm.pt')


def train_local(config):

    train_loader, valid_loader = cr.data_creator(config)
    model = cr.model_creator(config)
    loss_func = cr.loss_creator(config)
    optimizer = cr.optimizer_creator(model, config)
    #scheduler = cr.scheduler_creator(optimizer, config)

    gpu_available = config.get('gpu_available', False)
    batch_size = config.get('batch_size')

    # training params
    epochs = config.get('epochs', 4)
    counter = 0
    print_every = 100
    clip = 5 # gradient clipping

    # move model to GPU, if available
    if gpu_available:
        model.cuda()

    start_time = time.time()
    model.train()
    # train for some number of epochs
    for epoch in range(epochs):
        # initialize hidden state
        h = model.init_hidden(batch_size)

        # batch loop
        for inputs, labels in train_loader:
            counter += 1
            print('Epoch: {} Batch: {}'.format(epoch, counter))

            if gpu_available:
                inputs, labels = inputs.cuda(), labels.cuda()

            # Creating new variables for the hidden state, otherwise
            # we'd backprop through the entire training history
            h = tuple([each.data for each in h])

            # zero accumulated gradients
            model.zero_grad()

            # get the output from the model
            inputs = inputs.type(torch.LongTensor)
            output, h = model(inputs, h)

            # calculate the loss and perform backprop
            loss = loss_func(output.squeeze(), labels.float())
            loss.backward()
            # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
            nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()

            '''
            # loss stats
            if counter % print_every == 0:
                # Get validation loss
                val_h = model.init_hidden(batch_size)
                val_losses = []
                model.eval()
                for val_inputs, val_labels in valid_loader:

                    # Creating new variables for the hidden state, otherwise
                    # we'd backprop through the entire training history
                    val_h = tuple([each.data for each in val_h])

                    if gpu_available:
                        val_inputs, val_labels = val_inputs.cuda(), val_labels.cuda()

                    val_inputs = val_inputs.type(torch.LongTensor)
                    val_output, val_h = model(val_inputs, val_h)
                    val_loss = loss_func(val_output.squeeze(), val_labels.float())

                    val_losses.append(val_loss.item())

                model.train()
                print("Epoch: {}/{}...".format(epoch+1, epochs),
                    "Step: {}...".format(counter),
                    "Loss: {:.6f}...".format(loss.item()),
                    "Val Loss: {:.6f}".format(np.mean(val_losses)))
            '''

    duration = time.time() - start_time
    return model, duration


def train_distributed(config, num_workers=1, use_gpu=False):

    torch_trainer = TorchTrainer(
        training_operator_cls=SATrainingOperator,
        num_workers=num_workers,
        use_gpu=use_gpu,
        config=config,
        backend="gloo",
        scheduler_step_freq="epoch")

    epochs = config.get('epochs', 4)

    start_time = time.time()
    for i in range(epochs):
        stats = torch_trainer.train()
        print(stats)
    duration = time.time() - start_time

    #print(torch_trainer.validate())

    # If using Ray Client, make sure to force model onto CPU.
    model = torch_trainer.get_model(to_cpu=ray.util.client.ray.is_connected())
    print("Trained weight: % .5f, Bias: % .5f" % (
        model.weight.item(), model.bias.item()))
    torch_trainer.shutdown()
    return model, duration


def save_network(model):
    torch.save(model, NETWORK_FILE_PATH)


def load_network():
    model = torch.load(NETWORK_FILE_PATH)
    return model


def main(args):

    config = {
        'epochs': 4,
        'output_size': 1,
        'embedding_dim': 400,
        'hidden_dim': 256,
        'n_layers': 2,
        'lr': 0.001,
        'batch_size': 50,
        'gpu_available': torch.cuda.is_available()
    }

    # Loaders
    #batch_size = 50
    #train_loader, valid_loader = data_creator(config)

    # Network
    #gpu_available = torch.cuda.is_available()
    #net = network_creator(config)
    #print(net)
    #print(config)

    # Train
    if args.distribute:
        model, duration = train_distributed(config, num_workers=4)
    else:
        model, duration = train_local(config)
    print('Total elapsed training time: ', duration)
    print(type(model))
    save_network(model)


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
