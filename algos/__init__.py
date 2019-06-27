from pandas import DataFrame
from numpy import array
import numpy
import pandas


def shuffle_dataframe(df: DataFrame):
    return df.reindex(numpy.random.permutation(df.index))

def process_data_by_classification_and_set(df: DataFrame, training_set_size: int) -> tuple:
    # df: dataframe that has columns of x variables and a column of 'y', rows represent number of datapoints
    # return training set and testing set with different classification class binary result as dictionary of numpy arrays
    class_indices = pandas.unique(df['y'])
    df = shuffle_dataframe(df)
    result_dic = {}
    y_arrays_dic = {}
    result_dic['training_set'] = df.iloc[:training_set_size, :].copy()
    result_dic['testing_set'] = df.iloc[training_set_size:, :].copy()
    y_data = result_dic['training_set']['y'].copy()
    for class_index in class_indices:
        y_data_binary = y_data.where(y_data == class_index, other=0)
        y_data_binary = y_data_binary.where(y_data_binary == 0, other=1)
        y_arrays_dic[class_index] = array([y_data_binary.to_list()]).transpose()

    return result_dic, y_arrays_dic

class LogisticRegression(object):
    def __init__(self):
        pass

    def sigmoid_function(self, weights: array, x_variables: array) -> array:
        '''
        :param weights: shape of (number of features, 1) 2d array, cannot be 1d array with shape (number of weights, )
        :param x_variables: shape of (number of datapoints, number of features)
        :return: a 2d array with shape (number of datapoints, 1) representing probability of each class
        '''
        return 1 / (1 + numpy.exp(-(x_variables.dot(weights))))
