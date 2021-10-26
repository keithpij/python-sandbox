'''
Simple Pytorch example.
'''
import argparse
import time

import numpy as np
import ray
from ray.util.sgd import TorchTrainer
from ray.util.sgd.torch import TrainingOperator
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


def data_creator(config):
    '''
    Creates and returns a training dataloader and a validation dataloader.
    '''
    train_dataset = LinearDataset(2, 5, size=config.get("data_size", 1000))
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.get("batch_size", 32),
    )

    val_dataset = LinearDataset(2, 5, size=config.get("val_size", 400))
    validation_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config.get("batch_size", 32))

    return train_loader, validation_loader


def model_creator(config):
    """Returns a torch.nn.Module object."""
    return nn.Linear(1, config.get("hidden_size", 1))


def optimizer_creator(model, config):
    '''
    Returns optimizer defined upon the model parameters.
    '''
    return torch.optim.SGD(model.parameters(), lr=config.get("lr", 1e-2))


def scheduler_creator(optimizer, config):
    """Returns a learning rate scheduler wrapping the optimizer.

    You will need to set ``TorchTrainer(scheduler_step_freq="epoch")``
    for the scheduler to be incremented correctly.

    If using a scheduler for validation loss, be sure to call
    ``trainer.update_scheduler(validation_loss)``.
    """
    return torch.optim.lr_scheduler.StepLR(optimizer, step_size=config.get("step_size", 5), gamma=config.get("gamma", 0.9))


def loss_creator():
    return nn.MSELoss()  # this is for regression mean squared loss


def train_local(config):
    '''
    This function will train the model locally on a single CPU.
    '''
    train_loader, validation_loader = data_creator(config)
    model = model_creator(config)
    loss_func = loss_creator()
    optimizer = optimizer_creator(model, config)
    scheduler = scheduler_creator(optimizer, config)

    epochs = config.get('epochs', 4)
    start_time = time.time()
    for current_epoch in range(epochs):
        print('Epoch: ', current_epoch+1)
        print('Weight: % .5f, Bias: % .5f' % (model.weight.item(), model.bias.item()))
        print('Weight gradient:', model.weight.grad)
        print('Bias gradient: ', model.bias.grad)

        for step, (batch_x, batch_y) in enumerate(train_loader): # for each training step

            prediction = model(batch_x)     # input x and predict based on x

            loss = loss_func(prediction, batch_y)     # must be (1. nn output, 2. target)

            optimizer.zero_grad()   # clear gradients for next train
            loss.backward()         # backpropagation, compute gradients
            optimizer.step()        # apply gradients

        scheduler.step()
        print(f'Epoch {current_epoch+1}, Training loss: {loss.item():.4f}')
        print('\n')

    duration = time.time() - start_time
    print('Final weight: % .5f, Final bias: % .5f' % (model.weight.item(), model.bias.item()))

    return model, duration


def predict(model, predict_value):
    '''
    Make a prediction based on the model.
    '''
    tensor_input = torch.Tensor([float(predict_value)])
    print(tensor_input)

    output = model(tensor_input)
    print(output)


def main(config):
    '''
    Main CLI entry point.
    '''
    config = {
        "epochs": 5,
        "lr": 1e-2,  # used in optimizer_creator
        "hidden_size": 1,  # used in model_creator
        "batch_size": 4,  # used in data_creator
        "data_size": 10000,
        "val_size": 400,
        "gamma": 0.9,
        "step_size": 5
    }

    if args.distribute:
        model, duration = train_distributed(config, num_workers=4)
    else:
        model, duration = train_local(config)
    print('Total elapsed training time: ', duration)
    print(type(model))

    predict(model, args.predict_value)


def train_distributed(config, num_workers=1, use_gpu=False):

    CustomTrainingOperator = TrainingOperator.from_creators(
        model_creator=model_creator, optimizer_creator=optimizer_creator,
        data_creator=data_creator, scheduler_creator=scheduler_creator,
        loss_creator=nn.MSELoss)

    torch_trainer = TorchTrainer(
        training_operator_cls=CustomTrainingOperator,
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

    print(torch_trainer.validate())

    # If using Ray Client, make sure to force model onto CPU.
    model = torch_trainer.get_model(to_cpu=ray.util.client.ray.is_connected())
    print("Trained weight: % .5f, Bias: % .5f" % (
        model.weight.item(), model.bias.item()))
    torch_trainer.shutdown()
    return model, duration


if __name__ == "__main__":
    # Setup all the CLI arguments for this module.
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--predict_value',
                        help='Value for final prediction test.')
    parser.add_argument('-d', '--distribute',
                        help='Distribute the training task.',
                        action='store_true')

    # Parse what was passed in. This will also check the arguments for you and produce
    # a help message if something is wrong.
    args = parser.parse_args()
    main(args)
