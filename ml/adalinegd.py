'''
Adaptive Linear Neuron classifier
'''
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import common


class AdalineGD(object):
    '''
    Parameters
    eta : float - learning rate (between 0.0 and 1.0)
    iterations : int - the number of passes over the training dataset.

    Attributes
    weights : 1 dimensional array - the weights after fitting.
    errors : list - number of misclassifications in every epoch.
    '''


    def __init__(self, eta=0.01, iterations=10):
        self.eta = eta
        self.iterations = iterations
        self.cost = []


    def fit(self, feature_matrix, class_labels):
        '''
        Fit training data.

        Parameters

        feature_matrix : array-like, shape = [n_samples, n_features]
        Training vectors where n_samples is the number of samples and
        n_features is the number of features.

        class_labels : array-like, shape = [n_samples]
        Target values
        '''
        self.feature_matrix = feature_matrix
        self.class_labels = class_labels
        self.weights = np.zeros(feature_matrix.shape[1])
        self.threshold = 0

        for _ in range(self.iterations):
            output = self.net_input(self.feature_matrix)
            errors = self.class_labels - output
            self.weights += self.eta * self.feature_matrix.T.dot(errors)
            self.threshold += self.eta * errors.sum()
            cost = (errors**2).sum()/2.0
            self.cost.append(cost)

        return self


    def net_input(self, sample):
        ''' Calculate net input. '''
        return np.dot(sample, self.weights) + self.threshold


    def activation(self, sample):
        ''' Compute linear activation '''
        return self.net_input(sample)


    def predict(self, sample):
        ''' Return class label after unit step. '''
        # Returns 1 if net_input of sample is greater than 0.
        # Returns -1 otherwise.
        return np.where(self.activation(sample) >= 0.0, 1, -1)


def plot_cost_across_epochs():
    ''' Plot the cost across epochs. '''
    raw_data_frame = common.get_raw_data()
    feature_matrix, class_labels = common.data_preprocessing(raw_data_frame)

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(8,4))

    ada1 = AdalineGD(iterations=10, eta=0.01)
    ada1.fit(feature_matrix, class_labels)
    ada2 = AdalineGD(iterations=10, eta=0.0001)
    ada2.fit(feature_matrix, class_labels)

    ax[0].plot(range(1, len(ada1.cost)+1), np.log10(ada1.cost), marker='o')
    ax[0].set_xlabel('Epochs')
    ax[0].set_ylabel('log(Sum-squared-error)')
    ax[0].set_title('Adaline - Learning rate 0.01')

    ax[1].plot(range(1, len(ada2.cost)+1), np.log10(ada2.cost), marker='o')
    ax[1].set_xlabel('Epochs')
    ax[1].set_ylabel('log(Sum-squared-error)')
    ax[1].set_title('Adaline - Learning rate 0.0001')

    plt.show()


if __name__ == '__main__':
    plot_cost_across_epochs()
