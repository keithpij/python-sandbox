'''
Simply Ray SGD example.
'''

import argparse
import numpy as np
import torch
import torch.nn as nn

import ray
from ray.util.sgd import TorchTrainer
from ray.util.sgd.torch import TrainingOperator


class LinearDataset(torch.utils.data.Dataset):
    '''
    This class will emulate the function: y = a * x + b
    In machine learning terms a becomes our weight and b becomes the bias.
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
    """Returns training dataloader, validation dataloader."""
    train_dataset = LinearDataset(2, 5, size=config.get("data_size", 10000))
    val_dataset = LinearDataset(2, 5, size=config.get("val_size", 400))
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.get("batch_size", 32),
    )
    validation_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config.get("batch_size", 32))
    return train_loader, validation_loader


def train_example(num_workers=1, use_gpu=False):

    CustomTrainingOperator = TrainingOperator.from_creators(
        model_creator=model_creator, optimizer_creator=optimizer_creator,
        data_creator=data_creator, scheduler_creator=scheduler_creator,
        loss_creator=nn.MSELoss)

    torch_trainer = TorchTrainer(
        training_operator_cls=CustomTrainingOperator,
        num_workers=num_workers,
        use_gpu=use_gpu,
        config={
            "lr": 1e-2,  # used in optimizer_creator
            "hidden_size": 1,  # used in model_creator
            "batch_size": 4,  # used in data_creator
        },
        backend="gloo",
        scheduler_step_freq="epoch")

    for i in range(5):
        stats = torch_trainer.train()
        print(stats)

    print(torch_trainer.validate())

    # If using Ray Client, make sure to force model onto CPU.
    model = torch_trainer.get_model(to_cpu=ray.util.client.ray.is_connected())
    print("Trained weight: % .5f, Bias: % .5f" % (
        model.weight.item(), model.bias.item()))
    torch_trainer.shutdown()
    print(type(model))

    #input = torch.randn(1)
    #output = model(input)
    input = torch.Tensor([10])
    print(input)

    output = model(input)
    print(output)
    print('Model experiment completed sucessfully.')


if __name__ == "__main__":

    ray.init(num_cpus=2)
    train_example(num_workers=1, use_gpu=False)
