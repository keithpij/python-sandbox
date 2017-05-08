'''
Experiments with the Iris data set.
'''
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import perceptron


def get_raw_data():
    '''
    Get the iris data frame from the UCI Machine Learning Repository.
    '''
    #url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
    file = 'iris-dataset.csv'
    raw_data_frame = pd.read_csv(file, header=None)
    return raw_data_frame


def data_preprocessing(raw_data_frame):
    '''
    Create the class labels and the feature matrix from the raw data.
    Returns a feature matrix and the class labels.
    '''

    # The fourth column is the class label or the 'answer'.
    class_labels = raw_data_frame.iloc[0:100, 4].values

    # Convert 'Iris-setosa' to -1.
    # Convert 'Iris-virginica' to 1.
    class_labels = np.where(class_labels == 'I. setosa', -1, 1)

    # Get the feature columns for sepal length and petal length and
    # create a feature matrix.
    feature_matrix = raw_data_frame.iloc[0:100, [0, 2]].values
    return feature_matrix, class_labels


def plot_feature_matrix(feature_matrix):
    '''
    Plot the feature matrix as a scatter graph.
    '''
    plt.scatter(feature_matrix[:50, 0], feature_matrix[:50, 1],
                color='red', marker='o', label='setosa')

    plt.scatter(feature_matrix[50:100, 0], feature_matrix[50:100, 1],
                color='blue', marker='x', label='versicolor')

    plt.xlabel('sepal length')
    plt.ylabel('petal length')
    plt.legend(loc='upper left')
    plt.show()


def learn(feature_matrix, class_labels):
    classifier = perceptron.Perceptron(eta=0.1, iterations=10)
    classifier.fit(feature_matrix, class_labels)
    return classifier


def plot_epochs(classifier):
    plt.plot(range(1, len(classifier.errors) + 1), classifier.errors, marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Number of misclassifications')
    plt.show()


def plot_results(feature_matrix, class_labels, classifier, resolution=0.02):
    ''' Plot results '''
    markers = ('o', 'x', 's', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(class_labels))])

    # plot the decision surface
    x1_min, x1_max = feature_matrix[:, 0].min() - 1, feature_matrix[:, 0].max() + 1
    x2_min, x2_max = feature_matrix[:, 1].min() - 1, feature_matrix[:, 1].max() + 1

    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))

    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)

    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    # plot class samples
    for idx, class_label in enumerate(np.unique(class_labels)):
        plt.scatter(x=feature_matrix[class_labels == class_label, 0],
                    y=feature_matrix[class_labels == class_label, 1],
                    alpha=0.8, c=cmap(idx), marker=markers[idx], label=class_label)

    plt.xlabel('sepal length [cm]')
    plt.ylabel('petal length [cm]')
    plt.legend(loc='upper left')
    plt.show()


if __name__ == '__main__':
    RAW_DATA_FRAME = get_raw_data()
    #print(RAW_DATA_FRAME.tail())

    FEATURE_MATRIX, CLASS_LABELS = data_preprocessing(RAW_DATA_FRAME)
    #print(CLASS_LABELS)
    #print(FEATURE_MATRIX)
    CLASSIFIER = learn(FEATURE_MATRIX, CLASS_LABELS)

    #plot_feature_matrix(FEATURE_MATRIX)
    plot_epochs(CLASSIFIER)
    #plot_results(FEATURE_MATRIX, CLASS_LABELS, CLASSIFIER, 0.02)

    #for sample, target in zip(FEATURE_MATRIX, CLASS_LABELS):
    #    print(sample)
    #print(CLASSIFIER.weights)

    #unknown_vector = np.linalg.lstsq(FEATURE_MATRIX, CLASS_LABELS)
    #print(unknown_vector)

    #calculated_class_labels = np.dot(FEATURE_MATRIX, CLASSIFIER.weights[1:])
    #print(calculated_class_labels)
