'''
Perceptron classifier
'''
import csv
import numpy as np

IRIS_SETOSA = -1
IRIS_VIRGINICA = 1
TRACE_FILE = 'trace_file.csv'

class Perceptron(object):
    '''
    Percepton learning algorithm
    '''


    def __init__(self, feature_matrix, class_labels, eta=0.01, iterations=10):
        '''
        Parameters
        eta : float - learning rate (between 0.0 and 1.0)
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
        self.eta = eta
        self.iterations = iterations
        self.feature_matrix = feature_matrix
        self.class_labels = class_labels
        self.threshold = 0
        self.weights = np.zeros(feature_matrix.shape[1])
        self.errors = []


    def fit(self):
        '''
        Fit training data.
        '''

        with open('cfs.csv', 'w') as csvfile:
            
            # Header row in the csv file.
            fieldnames = ['Iteration','Weight for Sepal length','Sepal length', 'Weight for petal length','petal length',
                        'Threshold','Prediction', 'Label', 'Difference', 'eta', 'Update']
            writer = csv.DictWriter(TRACE_FILE, fieldnames=fieldnames)
            writer.writeheader()

            for iteration in range(self.iterations):
                errors = 0

                # The sample will be a vector containing the sepal length and the
                # petal length in centimeters.
                # The target will be either -1 (Iris-setosa) or 1 (Iris-virginica).
                sample_number = 0
                for sample, target in zip(self.feature_matrix, self.class_labels):

                    # Keeping track of sample number for logging purposes.
                    sample_number += 1

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

                    if sample_number == 1:
                        to_csv = dict()
                        to_csv['Iteration'] = iteration
                        to_csv['Weight for Sepal length'] = self.weights[0]
                        to_csv['Sepal length'] = sample[0]
                        to_csv['Weight for petal length'] = self.weights[1]
                        to_csv['Petal length'] = sample[1]
                        to_csv['Threshold'] = self.threshold
                        to_csv['Prediction'] = prediction
                        to_csv['Label'] = target
                        to_csv['Difference'] = difference
                        to_csv['eta'] = self.eta
                        to_csv['Update'] = update
                        writer.writerow(to_csv)

            self.errors.append(errors)


    def net_input(self, sample):
        ''' Calculate net input.
        '''
        return np.dot(sample, self.weights) + self.threshold


    def predict(self, sample):
        ''' Return class label after unit step. '''
        # Returns 1 if net_input of sample is greater than 0.
        # Returns -1 otherwise.
        return np.where(self.net_input(sample) >= 0.0, IRIS_VIRGINICA, IRIS_SETOSA)
