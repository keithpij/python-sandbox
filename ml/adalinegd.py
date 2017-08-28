'''
Adaptive Linear Neuron classifier
'''
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import common

IRIS_SETOSA = -1
IRIS_VERSICOLOR = 1


class AdalineGD(object):
    '''
    Parameters
    eta : float - learning rate (between 0.0 and 1.0)
    iterations : int - the number of passes over the training dataset.

    Attributes
    weights : 1 dimensional array - the weights after fitting.
    errors : list - number of misclassifications in every epoch.
    '''


    def __init__(self, feature_matrix, class_labels, learning_rate=0.01, iterations=10):
        '''
        Parameters
        learning_rate : float - learning rate (between 0.0 and 1.0)
        iterations : int - the number of passes over the training dataset.

        feature_matrix : matrix, shape = [n_samples, n_features]
        Training vectors where n_samples is the number of samples and
        n_features is the number of features.

        class_labels : vector, shape = [n_samples]
        Target values

        Attributes
        weights : 1 dimensional array - the weights after fitting.
        errors : list - number of misclassifications in every epoch.
        '''
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.feature_matrix = feature_matrix
        self.class_labels = class_labels
        self.threshold = 0
        self.weights = np.zeros(feature_matrix.shape[1])
        self.cost = []


    def fit(self):
        '''
        Fit training data.
        '''

        for _ in range(self.iterations):
            output = self.net_input(self.feature_matrix)
            errors = self.class_labels - output
            self.weights += self.learning_rate * self.feature_matrix.T.dot(errors)
            self.threshold += self.learning_rate * errors.sum()
            cost = (errors**2).sum()/2.0
            self.cost.append(cost)


    def net_input(self, sample):
        ''' Calculate net input. '''
        return np.dot(sample, self.weights) + self.threshold


    def activation(self, sample):
        ''' Compute linear activation '''
        return self.net_input(sample)


    def predict(self, sample):
        '''
        Return class label based on a unit step function.
        Returns 1 (for Iris Versicolor) if net_input of sample is greater than 0.
        Returns -1 (for Iris Setosa) otherwise.
        This function (like the net_input function) can be called with a single sample or an
        entire feature matrix.
        '''
        return np.where(self.activation(sample) >= 0.0, IRIS_VERSICOLOR, IRIS_SETOSA)


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
