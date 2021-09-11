'''
Simply Pytorch example.
'''

import argparse
import numpy as np
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


def model_creator(config):
    """Returns a torch.nn.Module object."""
    return nn.Linear(1, config.get("hidden_size", 1))


def optimizer_creator(model, config):
    """Returns optimizer defined upon the model parameters."""
    return torch.optim.SGD(model.parameters(), lr=config.get("lr", 1e-2))


def scheduler_creator(optimizer, config):
    """Returns a learning rate scheduler wrapping the optimizer.

    You will need to set ``TorchTrainer(scheduler_step_freq="epoch")``
    for the scheduler to be incremented correctly.

    If using a scheduler for validation loss, be sure to call
    ``trainer.update_scheduler(validation_loss)``.
    """
    return torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.9)


def data_creator(config):
    '''
    Creates and returns a training dataloader and a validation dataloader.
    '''
    train_dataset = LinearDataset(2, 5, size=config.get("data_size", 10000))
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.get("batch_size", 32),
    )

    val_dataset = LinearDataset(2, 5, size=config.get("val_size", 400))
    validation_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config.get("batch_size", 32))

    return train_loader, validation_loader


def train(model, optimizer, scheduler, loader, epoch=5):
    '''
    This function will train the model.
    '''
    
    loss_func = nn.MSELoss()  # this is for regression mean squared loss

    for e in range(epoch):
        print('Epoch: ', e+1)
        print('Weight: % .5f, Bias: % .5f' % (model.weight.item(), model.bias.item()))
        print('Weight gradient:', model.weight.grad)
        print('Bias gradient: ', model.bias.grad)

        for step, (batch_x, batch_y) in enumerate(loader): # for each training step

            prediction = model(batch_x)     # input x and predict based on x

            loss = loss_func(prediction, batch_y)     # must be (1. nn output, 2. target)
            #print(loss)

            optimizer.zero_grad()   # clear gradients for next train
            loss.backward()         # backpropagation, compute gradients
            optimizer.step()        # apply gradients

            #print('Weight: % .5f, Bias: % .5f' % (model.weight.item(), model.bias.item()))
            #print('Weight gradient: % .5f, Bias gradient: % .5f' % (model.weight.grad, model.bias.grad))
            #print('Weight gradient:', model.weight.grad)
            #print('Bias gradient: ', model.bias.grad)
            #print('\n')

            #bra

        scheduler.step()
        print(f'Epoch {e+1}, Training loss: {loss.item():.4f}')
        print('\n')

    print('Final weight: % .5f, Final bias: % .5f' % (model.weight.item(), model.bias.item()))
    print(type(model))

    input = torch.Tensor([10])
    print(input)

    output = model(input)
    print(output)


if __name__ == "__main__":

    config = {
        "lr": 1e-2,  # used in optimizer_creator
        "hidden_size": 1,  # used in model_creator
        "batch_size": 4,  # used in data_creator
    }
    train_loader, validation_loader = data_creator(config)
    model = model_creator(config)
    optimizer = optimizer_creator(model, config)
    scheduler = scheduler_creator(optimizer, config)
    train(model, optimizer, scheduler, train_loader, 5)
