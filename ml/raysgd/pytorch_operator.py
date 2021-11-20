'''
This module contains the Sentiment Analysis Training Operator.
'''
from ray.util.sgd.torch import TrainingOperator
import torch
import torch.nn as nn

import pytorch_creators as cr


class SATrainingOperator(TrainingOperator):
    '''
    Training operator for the Sentiment Analysis model.
    '''
    def setup(self, config):
        # Need to register the data first since the data_creator adds
        # config needed by the model (network).
        train_loader, val_loader = cr.data_creator(config)
        self.register_data(train_loader=train_loader, validation_loader=val_loader)

        # Register the model, loss, and optimizer.
        model = cr.model_creator(config)
        loss = cr.loss_creator(config)
        optimizer = cr.optimizer_creator(model, config)
        self.model, self.optimizer, self.criterion = self.register(
                models=model, optimizers=optimizer, criterion=loss)

        # Setup the initial hidden state.
        batch_size = self.config.get('batch_size')
        gpu_available = self.config.get('gpu_available')
        self.hidden_initial = model.init_hidden(batch_size, gpu_available)
        #print('Traing operator created.')

    def train_epoch(self, iterator, info):
        #print('Start of train_epoch.')

        gpu_available = self.config.get('gpu_available')
        clip = 5 # gradient clipping

        # initialize hidden state
        h = self.hidden_initial

        counter = 0
        # batch loop
        for inputs, labels in iterator:
            counter += 1
            #print('Batch: {}'.format(counter))

            if gpu_available:
                inputs, labels = inputs.cuda(), labels.cuda()

            # Creating new variables for the hidden state, otherwise
            # we'd backprop through the entire training history
            h = tuple([each.data for each in h])

            # zero accumulated gradients
            self.model.zero_grad()

            # get the output from the model
            inputs = inputs.type(torch.LongTensor)
            output, h = self.model(inputs, h)

            # calculate the loss and perform backprop
            loss = self.criterion(output.squeeze(), labels.float())
            loss.backward()
            # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
            nn.utils.clip_grad_norm_(self.model.parameters(), clip)
            self.optimizer.step()


        # Returns stats of the meters.
        metrics = {"info": info} # dict of metrics
        return metrics