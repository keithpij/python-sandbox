'''
Perceptron classifier
'''
import numpy as np

class Perceptron(object):
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


    def fit(self, x, y):
        '''
        Fit training data.

        Parameters

        x : array-like, shape = [n_samples, n_features]
        Training vectors where n_samples is the number of samples and
        n_features is the number of features.

        y : array-like, shape = [n_samples]
        Target values

        returns

        self : object
        '''
        self.weights = np.zeros(1 + x.shape[1])
        self.errors = []

        for _ in range(self.iterations):
            errors = 0
            for xi, target in zip(x,y):
                update = self.eta * (target - self.predict(xi))
                self.weights[1:] += update * xi
                self.weights[0] += update
                errors += int(update != 0.0)
            self.errors.append(errors)

        return self


    def net_input(self, x):
        # Calculate net input
        return np.dot(x, self.weights[1:]) + self.weights[0]


    def predict(self, x):
        # Return class label after unit step.
        return np.where(self.net_input(x) >= 0.0, 1, -1)

