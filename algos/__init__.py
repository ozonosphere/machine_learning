from pandas import DataFrame
from numpy import array
import numpy
import pandas
from data import get_data_from_mat
numpy.seterr(divide = 'ignore')


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

def convert_data_to_dict_of_array(data_tuple: tuple) -> dict:
    training_data_set = data_tuple[0]['training_set']
    testing_data_set = data_tuple[0]['testing_set']
    y_variables = training_data_set['y'].values
    training_y_variable_arrays = data_tuple[1]
    training_x_variable_array = training_data_set.values[:, :-1]
    testing_x_variable_array = testing_data_set.values[:, :-1]
    testing_y_variables = testing_data_set['y'].values
    return {'x_variables': training_x_variable_array, 'y_variables': y_variables,
            'y_variables_by_class': training_y_variable_arrays, 'testing_x_variables': testing_x_variable_array,
            'testing_y_variables': testing_y_variables}

class LogisticRegression(object):
    # This Regression is specifically designed for classification as the cost function is the log function (convex function)
    # The cost function can be others to solve different problems
    def __init__(self, x_variables, y_variables, y_variables_by_class, learning_rate, testing_x_variables=None, testing_y_variables=None,
                 regularized_lambda=0.0, number_of_iteration=1000):
        self.y_variables_by_class = y_variables_by_class # dictionary of class index with corresponding y_variables array (binary, 1 and 0)
        self.number_of_class = len(self.y_variables_by_class)
        self.x_variables = x_variables
        self.y_variables = y_variables # 1d array (number of datapoints, )
        self.testing_x_variables = testing_x_variables
        self.testing_y_variables = testing_y_variables
        self.regularized_lambda = regularized_lambda
        self.learning_rate = learning_rate
        self.number_of_iteration = number_of_iteration
        self.iteration_cost_df_by_class = {}

    def initialising_weights(self):
        self.optimal_weights_dict_by_class = {}
        weight_vector_shape = self.x_variables[0, :].shape # shape (number of features, )
        for class_ in self.y_variables_by_class:
            self.optimal_weights_dict_by_class[class_] = numpy.random.uniform(size=(weight_vector_shape[0], 1))


    def eval_sigmoid_function(self, x_variables, weights: array) -> array:
        '''
        :param weights: shape of (number of features, 1) 2d array, cannot be 1d array with shape (number of features, )
        :param x_variables: shape of (number of datapoints, number of features)
        :return: a 2d array with shape (number of datapoints, 1) representing probability of each class
        '''
        return 1 / (1 + numpy.exp(-(x_variables.dot(weights))))

    def eval_cost_function(self, weights: array, y_variables: array) -> float:
        probability_of_each_sample = self.eval_sigmoid_function(self.x_variables, weights) # shape (number of datapoints, 1)
        a = numpy.log(1 - probability_of_each_sample)
        numpy.place(a, a == -numpy.inf, 0.0)
        b = numpy.log(probability_of_each_sample)
        numpy.place(b, b == -numpy.inf, 0.0)
        total_costs = -y_variables * b - (1 - y_variables) * a
        number_of_datapoints = len(total_costs)
        # if regularized_lambda is present, the weight of first feature is used as bias term.
        result = numpy.mean(total_costs) + self.regularized_lambda / (2 * number_of_datapoints) * numpy.sum(numpy.square(weights[1:, :]))
        return result

    def eval_cost_derivatives_as_weights(self, weights: array, y_variables: array) -> array:
        """
        :return: weights partial derivatives with shape (number of features, 1)
        """
        costs = self.eval_sigmoid_function(self.x_variables, weights)
        number_of_datapoints = len(y_variables)
        derivatives = 1 / number_of_datapoints * (self.x_variables.transpose().dot((costs - y_variables))) # shape (number of features, 1)
        regularized_terms = self.regularized_lambda / number_of_datapoints * weights
        regularized_terms[0, :] = 0 # first feature weight is bias term if regularized_lambda is given
        derivatives = derivatives + regularized_terms
        return derivatives

    def get_optimal_weights_by_gradient_decent(self):
        for class_ in self.y_variables_by_class:
            weights = self.optimal_weights_dict_by_class[class_]
            y_variables = self.y_variables_by_class[class_]
            iteration_list = []
            for i in range(1, self.number_of_iteration + 1):
                cost = self.eval_cost_function(weights, y_variables)
                weights = weights - self.learning_rate * self.eval_cost_derivatives_as_weights(weights, y_variables)
                self.optimal_weights_dict_by_class[class_] = weights
                iteration_list.append({'No_of_iteration': i, 'cost': cost})
            self.iteration_cost_df_by_class[class_] = DataFrame(iteration_list)

    def get_prob_for_each_class_by_optimal_weights(self, x_variables) -> DataFrame:
        probability_by_class = {}
        for class_, weights in self.optimal_weights_dict_by_class.items():
            probability_class = self.eval_sigmoid_function(x_variables, weights)
            shape = (probability_class.shape[0], )
            probability_class = probability_class.reshape(shape)
            probability_by_class[class_] = probability_class
        return DataFrame(probability_by_class)

    def train(self):
        self.initialising_weights()
        self.get_optimal_weights_by_gradient_decent()
        prob = self.get_prob_for_each_class_by_optimal_weights(self.x_variables)
        self.training_probability_by_class_index_df = prob
        self.training_predicted_y = self.training_probability_by_class_index_df.idxmax(axis='columns').values
        self.training_accuracy = numpy.sum(self.y_variables == self.training_predicted_y) / len(self.y_variables)
        print('Training accuracy is: ' + str(self.training_accuracy))

    def test(self):
        if self.testing_x_variables is None or self.testing_y_variables is None:
            raise IndexError('No testing data found!')
        else:
            prob = self.get_prob_for_each_class_by_optimal_weights(self.testing_x_variables)
            self.testing_probability_by_class_index_df = prob
            self.testing_predicted_y = self.testing_probability_by_class_index_df.idxmax(axis='columns').values
            self.testing_accuracy = numpy.sum(self.testing_y_variables == self.testing_predicted_y) / len(self.testing_y_variables)
            print('Testing accuracy is: ' + str(self.testing_accuracy))


if __name__ == '__main__':
    training_set_lenth = 4000
    original_data = get_data_from_mat('ex3data1.mat')
    x_data = DataFrame(original_data['X'])
    y_data = DataFrame(original_data['y'])
    y_data.columns = ['y']
    xy_data_df = pandas.concat([x_data, y_data], axis='columns')
    data_store_dictionary = process_data_by_classification_and_set(xy_data_df, training_set_lenth)
    input_data_dictionary = convert_data_to_dict_of_array(data_store_dictionary)

    input_data_dictionary['regularized_lambda'] = 0.0
    input_data_dictionary['number_of_iteration'] = 1000
    input_data_dictionary['learning_rate'] = 0.01
    log_reg = LogisticRegression(**input_data_dictionary)
    log_reg.train()