'''
Common code for loading and processing the Iris data set.
'''
import pandas as pd
import numpy as np


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
