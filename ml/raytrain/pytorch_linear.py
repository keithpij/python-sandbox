'''
Simple Pytorch example.
'''
import argparse
import os
import time
from urllib import response

import numpy as np
import ray
import ray.train as train
from ray.train import Trainer
from ray.train.callbacks import JsonLoggerCallback, TBXLoggerCallback
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DistributedSampler
import torch
import torch.nn as nn


class LinearDataset(torch.utils.data.Dataset):
    '''
    This class will emulate the function: y = m * x + b
    In machine learning terms m becomes our weight and b becomes the bias.
    '''

    def __init__(self, a, b, size=1000):
        x = np.arange(0, 10, 10 / size, dtype=np.float32)
        self.x = torch.from_numpy(x)
        self.y = torch.from_numpy(a * x + b)

    def __getitem__(self, index):
        return self.x[index, None], self.y[index, None]

    def __len__(self):
        return len(self.x)


def train_batches(dataloader, model, loss_fn, optimizer):
    for X, y in dataloader:
        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def validate_epoch(dataloader, model, loss_fn):
    num_batches = len(dataloader)
    model.eval()
    loss = 0
    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            loss += loss_fn(pred, y).item()
    loss /= num_batches
    result = {'process_id': os.getpid(), 'loss': loss}
    return result


def training_setup(config):
    '''
    This function will the datasets, model, loss function, and optimzer.
    '''
    data_size = config.get("data_size", 1000)
    val_size = config.get("val_size", 400)
    hidden_size = config.get("hidden_size", 1)
    lr = config.get("lr", 1e-2)

    train_dataset = LinearDataset(2, 5, size=data_size)
    val_dataset = LinearDataset(2, 5, size=val_size)

    model = nn.Linear(1, hidden_size)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    return train_dataset, val_dataset, model, loss_fn, optimizer


def train_epochs_local(config):
    start_time = time.time()

    train_dataset, val_dataset, model, loss_fn, optimizer = training_setup(config)

    batch_size = config.get("batch_size", 32)
    epochs = config.get("epochs", 3)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size)

    validation_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size)

    results = []
    for epoch in range(epochs):
        train_batches(train_loader, model, loss_fn, optimizer)
        result = validate_epoch(validation_loader, model, loss_fn)
        result['epoch'] = epoch + 1
        results.append(result)

    duration = time.time() - start_time
    return model.state_dict(), results, duration


def train_epochs_remote(config):
    '''
    This function will be run on a remote worker.
    '''
    train_dataset, val_dataset, model, loss_fn, optimizer = training_setup(config)

    batch_size = config.get("batch_size", 32)
    epochs = config.get("epochs", 3)

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

    results = []
    for epoch in range(epochs):
        train_batches(train_loader, model, loss_fn, optimizer)
        result = validate_epoch(validation_loader, model, loss_fn)
        result['epoch'] = epoch + 1
        train.report(**result)
        results.append(result)

    return model.state_dict(), results


def start_ray_train(config, num_workers=1):
    '''
    This function will setup the Ray Trainer which in turn will create the specified
    number of workers.
    Once the Trainer is created it is run and send the train_remote_epochs function.
    '''
    ray.init(num_cpus=4)
    trainer = Trainer(backend="torch", num_workers=num_workers)
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
    hidden_size = config.get("hidden_size", 1)
    model = nn.Linear(1, hidden_size)
    model.load_state_dict(torch.load(file_path))
    return model


def predict(model, input):
    '''
    Make a prediction based on the model.
    '''
    tensor_input = torch.Tensor([float(input)])
    print(tensor_input)

    output = model(tensor_input)
    print(output)


def main(args):
    '''
    Main CLI entry point.
    '''
    config = {
        "epochs": 5,
        "lr": 1e-2,  # Used by the optimizer.
        "hidden_size": 1,  # Used by the model.
        "batch_size": 1000,  # How the data is chunked up for training.
        "data_size": 1000000,
        "val_size": 400
    }

    if args.predict:
        model = load_model(args.model, config)
        predict(model, args.predict)
        return

    if args.distribute:
        model_state_dict, results, duration = start_ray_train(config, num_workers=4)
        save_model(model_state_dict, 'linear_distributed.pt')

    if args.local:
        model_state_dict, results, duration = train_epochs_local(config)
        save_model(model_state_dict, 'linear_local.pt')

    print('Total elapsed training time: ', duration)
    
    if args.verbose:
        print(results)


if __name__ == "__main__":
    # Setup all the CLI arguments for this module.
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
                        help='Make a prediction using a pre-trained model.')
    parser.add_argument('-v', '--verbose',
                        help='Verbose output (show results list).',
                        action='store_true')

    # Parse what was passed in. This will also check the arguments for you and produce
    # a help message if something is wrong.
    args = parser.parse_args()
    main(args)
