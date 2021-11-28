'''
Simple Pytorch example.
'''
import argparse
import time

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


def train_func(dataloader, model, loss_fn, optimizer):
    for X, y in dataloader:
        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def validate(dataloader, model, loss_fn):
    num_batches = len(dataloader)
    model.eval()
    loss = 0
    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            loss += loss_fn(pred, y).item()
    loss /= num_batches
    result = {"model": model.state_dict(), "loss": loss}
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


def train_local(config):
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
    for _ in range(epochs):
        train_func(train_loader, model, loss_fn, optimizer)
        result = validate(validation_loader, model, loss_fn)
        results.append(result)

    duration = time.time() - start_time
    return duration, results


def train_remote(config):
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

    # Create a new model for distributed training.
    model = DistributedDataParallel(model)

    results = []
    for _ in range(epochs):
        train_func(train_loader, model, loss_fn, optimizer)
        result = validate(validation_loader, model, loss_fn)
        train.report(**result)
        results.append(result)

    #torch.save(model, 'linear.pt')
    return results


def train_distributed(config, num_workers=1):

    ray.init(num_cpus=4)
    trainer = Trainer(backend="torch", num_workers=num_workers)
    trainer.start()

    start_time = time.time()
    results = trainer.run(
        train_remote,
        config)

    duration = time.time() - start_time

    trainer.shutdown()
    
    #model = torch.load('linear.pt')
    #print(type(model))

    return duration, results 


def predict(model, predict_value):
    '''
    Make a prediction based on the model.
    '''
    tensor_input = torch.Tensor([float(predict_value)])
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
        "batch_size": 10000,  # How the data is chunked up for training.
        "data_size": 1000000,
        "val_size": 400
    }

    if args.distribute:
        duration, results = train_distributed(config, num_workers=4)
    else:
        duration, results = train_local(config)
    
    print('Total elapsed training time: ', duration)
    
    if args.verbose:
        print(results)

    #if args.predict:
    #    predict(model, args.predict_value)


if __name__ == "__main__":
    # Setup all the CLI arguments for this module.
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--predict_value',
                        help='Value for final prediction test.')
    parser.add_argument('-d', '--distribute',
                        help='Distribute the training task.',
                        action='store_true')
    parser.add_argument('-v', '--verbose',
                        help='Verbose output (show results list).',
                        action='store_true')

    # Parse what was passed in. This will also check the arguments for you and produce
    # a help message if something is wrong.
    args = parser.parse_args()
    main(args)
