from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import Tensor


def plot_data(x: np.ndarray, y1: np.ndarray, y2: np.ndarray=None) -> None:
    '''
    Function for plotting X, y, and predictions.
    '''
    ax = plt.subplots()[1]
    ax.set_xlim(x.min()-5, x.max()+5)
    ax.set_ylim(y1.min()-5, y1.max()+5)
    plt.scatter(x, y1, color='blue', marker='s')
    if not y2 is None:
        ax.scatter(x, y2, color='red')
    plt.grid(True)
    plt.axhline(color='black')
    plt.axvline(color='black')


def print_results(model: nn.Module, losses: list) -> None:
    '''
    Function for printing training results
    '''
    epochs = len(losses)
    # Print the losses of every 10th epoch.
    for epoch in range(0, epochs, 10):
        print(epoch, ':', losses[epoch])
    # This will print the very last epoch so we can see the
    # final loss value.
    print(epochs-1, ':', losses[epochs-1])


def print_parameters(model: nn.Module) -> None:
    for name, parameter in model.named_parameters():
        print(name, parameter.data)
    

def train_model(model: nn.Module, config, X_train, y_train) -> Tuple[nn.Module, List]:
    epochs = config['epochs']
    lr = config['lr']
    loss_function = config['loss_function']

    torch.manual_seed(42)

    losses = []
    optimizer = optim.SGD(model.parameters(), lr=lr)

    for epoch in range(epochs):
        total_loss = 0
        for X, y in zip(X_train, y_train):

            # Wrap in tensors
            X_tensor = torch.from_numpy(X)
            y_tensor = torch.from_numpy(y)
            
            # Pytorch accumulates gradients so before passing in a new
            # context (features) you need to zero out the gradients from the 
            # previous context.
            model.zero_grad()
            optimizer.zero_grad()

            # Forward pass - this will get log probabilities for every word 
            # in our vocabulary which is now represented as embeddings.
            prediction = model(X_tensor)

            # Compute the loss.
            # target has to be a list for some reason.
            loss = loss_function(prediction, y_tensor)

            # Backward pass to update the gradients.
            loss.backward()

            # Optimize the parameters
            optimizer.step()

            # Get the loss for this context.
            total_loss += loss.item()
            
        # Save the total loss for this epoch.
        losses.append(total_loss)

    return model, losses