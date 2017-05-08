'''
Adaptive Linear Neuron classifier
'''
import numpy as np

class AdalineSGD(object):
    '''
    Parameters
    eta : float - learning rate (between 0.0 and 1.0)
    iterations : int - the number of passes over the training dataset.

    Attributes
    weights : 1 dimensional array - the weights after fitting.
    errors : list - number of misclassifications in every epoch.
    shuffle : bool (default = True) - if True shuffles training data every epoch to prevent cycles.
    random_state : int (default = None) - set random state for shuffling and initializing the weights.
    '''


    def __init__(self, eta=0.01, iterations=10, shuffle=True, random_state=None):
        self.eta = eta
        self.iterations = iterations
        self.shuffle = shuffle
        self.random_state = random_state
        if random_state:
            seed(random_state)
        self.weights_initialized = False


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

        self.initialize_weights()
        self.cost = []

        for i in range(self.iterations):
            if self.shuffle:
                feature_matrix, class_labels = self.shuffle_training_data(feature_matrix, class_labels)
            cost = []

            # sample will be a vector containing the sepal length and the
            # petal length in centimeters.
            # target will be either -1 (Iris-setosa) or 1 (Iris-virginica).
            for sample, target in zip(feature_matrix, class_labels):
                cost.append(self.update_weights(sample, target))
            average_cost = sum(cost)/len(class_labels)
            self.cost.append(average_cost)


    def shuffle_training_data(self, feature_matrix, class_labels):
        '''
        Shuffles the feature matrix and class labels.
        The permutation function returns an array of ints randomly ordered.
        '''
        random_indecies = np.random.permutation(len(class_labels))
        return feature_matrix[random_indecies], class_labels[random_indecies]


    def initialize_weights(self):
        ''' Initialize the weights and the threshold. '''
        self.threshold = 0
        self.weights = np.zeros(feature_matrix.shape[1])
        self.weights_initialized = True


    def update_weights(self, sample, target):
        ''' Apply the Adaline learning rule to update the weights. '''
        output = self.net_input(sample)
        error = (target - output)
        self.weights += self.eta * sample.dot(error)
        self.threshold += self.eta * error
        cost = 0.5 * error**2
        return cost


    def net_input(self, sample):
        ''' Calculate net input. '''
        return np.dot(sample, self.weights) + self.threshold
        

    def predict(self, sample):
        ''' Return class label after unit step. '''
        # Returns 1 if net_input of sample is greater than 0.
        # Returns -1 otherwise.
        return np.where(self.net_input(sample) >= 0.0, 1, -1)

