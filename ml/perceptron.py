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
        self.threshold = 0
        self.weights = np.zeros(feature_matrix.shape[1])
        self.errors = []

        for _ in range(self.iterations):
            errors = 0

            # sample will be a vector containing the sepal length and the
            # petal length in centimeters.
            # target will be either -1 (Iris-setosa) or 1 (Iris-virginica).
            for sample, target in zip(feature_matrix, class_labels):
                # prediction will be -1 or 1
                prediction = self.predict(sample)

                # If target and prediction are the same then
                # the difference is 0.
                # Otherwise difference will be 2 or -2.
                difference = target - prediction

                # Apply the learning rate which is a percentage of the difference.
                update = self.eta * difference

                self.weights += update * sample
                self.threshold += update

                errors += int(update != 0.0)

            self.errors.append(errors)


    def net_input(self, sample):
        ''' Calculate net input. '''
        return np.dot(sample, self.weights) + self.threshold


    def predict(self, sample):
        ''' Return class label after unit step. '''
        # Returns 1 if net_input of sample is greater than 0.
        # Returns -1 otherwise.
        return np.where(self.net_input(sample) >= 0.0, 1, -1)

