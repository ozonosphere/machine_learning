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
    return {'training_x_variables': training_x_variable_array, 'training_y_variables': y_variables,
            'y_variables_by_class': training_y_variable_arrays, 'testing_x_variables': testing_x_variable_array,
            'testing_y_variables': testing_y_variables}

class MachineLearningAlgos(object):
    def __init__(self):
        pass

class LogisticRegression(MachineLearningAlgos):
    # This Regression is specifically designed for classification as the cost function is the log function (convex function)
    # The cost function can be others to solve different problems
    def __init__(self, training_x_variables: array, training_y_variables: array, y_variables_by_class: dict, learning_rate: float, testing_x_variables: array=None, testing_y_variables: array=None,
                 regularized_lambda: float=0.0, number_of_iteration: int=1000, **unused):
        super().__init__()
        self.y_variables_by_class = y_variables_by_class # dictionary of class index with corresponding y_variables array (binary, 1 and 0)
        self.number_of_class = len(self.y_variables_by_class)
        self.training_x_variables = training_x_variables
        self.training_y_variables = training_y_variables # 1d array (number of datapoints, )
        self.testing_x_variables = testing_x_variables
        self.testing_y_variables = testing_y_variables
        self.regularized_lambda = regularized_lambda
        self.learning_rate = learning_rate
        self.number_of_iteration = number_of_iteration
        self.iteration_cost_df_by_class = {}

    def initialising_weights(self):
        self.optimal_weights_dict_by_class = {}
        weight_vector_shape = self.training_x_variables[0, :].shape # shape (number of features, )
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
        probability_of_each_sample = self.eval_sigmoid_function(self.training_x_variables, weights) # shape (number of datapoints, 1)
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
        costs = self.eval_sigmoid_function(self.training_x_variables, weights)
        number_of_datapoints = len(y_variables)
        derivatives = 1 / number_of_datapoints * (self.training_x_variables.transpose().dot((costs - y_variables))) # shape (number of features, 1)
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
        prob = self.get_prob_for_each_class_by_optimal_weights(self.training_x_variables)
        self.training_probability_by_class_index_df = prob
        self.training_predicted_y = self.training_probability_by_class_index_df.idxmax(axis='columns').values
        self.training_accuracy = numpy.sum(self.training_y_variables == self.training_predicted_y) / len(self.training_y_variables)
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


class NeuralNetwork(MachineLearningAlgos):
    def __init__(self, layer_structure: dict, training_x_variables: array,
                 training_y_variables: array, learning_rate: float, no_of_training_batches: int=1, testing_x_variables: array=None,
                 testing_y_variables: array=None, validation_x_variables: array=None, validation_y_variables: array=None,
                 regularized_lambda: float=0.0, cost_function: str='log',
                 number_of_iteration: int=1000, **unused):
        super().__init__()
        self.no_of_training_batches = no_of_training_batches
        self.number_of_output_classes = len(numpy.unique(training_y_variables))
        self.number_of_layers = max(layer_structure.keys())
        self.layer_structure = layer_structure # structure of {layer number: number of nodes (excluding a(l, 0), ...}
        self.training_x_variables = training_x_variables
        self.training_y_variables = training_y_variables  # 1d array (number of datapoints, )
        self.layer_structure[1] = self.training_x_variables.shape[1]
        self.layer_structure[self.number_of_layers] = self.number_of_output_classes
        self.testing_x_variables = testing_x_variables
        self.testing_y_variables = testing_y_variables
        self.regularized_lambda = regularized_lambda
        self.learning_rate = learning_rate
        self.cost_function = cost_function
        self.optimal_weights_by_layer_number = {}
        # Activation values by layer number with bias appended
        self.activation_values_by_layer_number = {}
        self.activation_values_including_ones_by_layer_number = {}
        self.error_by_layer_number = {}
        self.derivatives_of_cost_vs_weights_by_layer = {}
        self.activation_derivatives_by_layer_number = {} # Derivatives of activation against z where z = wx (including bias)
        self.number_of_iteration = number_of_iteration
        self.slice_training_data_into_batches()
        self.initializing_weights()

    def slice_training_data_into_batches(self):
        self.input_layer_x_variables_by_batch = {}
        self.output_layer_y_variables_by_batch = {}
        x_variable_batches = numpy.array_split(self.training_x_variables, self.no_of_training_batches)
        y_variable_batches = numpy.array_split(self.training_y_variables, self.no_of_training_batches)
        for batch_number in range(self.no_of_training_batches):
            self.input_layer_x_variables_by_batch[batch_number + 1] = x_variable_batches[batch_number]
            self.output_layer_y_variables_by_batch[batch_number + 1] = y_variable_batches[batch_number]

    def append_ones_to_activation(self, x_variables: array):
        self.activation_values_including_ones_by_layer_number[1] = self.append_bias_ones_to_array(x_variables)
        self.activation_values_by_layer_number[1] = x_variables

    def append_bias_ones_to_array(self, array):
        # array with shape (number of datapoints, number of features)
        ones_array = numpy.ones((array.shape[0], 1))
        return numpy.append(array[:, ::-1], ones_array, axis=1)[:, ::-1]

    def initializing_weights(self):
        for layer_number, no_of_nodes in self.layer_structure.items():
            if layer_number == 1:
                pass
            else:
                no_of_nodes_previous_layer = self.layer_structure[layer_number - 1]
                range_of_rand = 6 ** 0.5 / (no_of_nodes + no_of_nodes_previous_layer) ** 0.5
                self.optimal_weights_by_layer_number[layer_number] = numpy.random.uniform(-range_of_rand, range_of_rand, size=(no_of_nodes, no_of_nodes_previous_layer + 1))
                self.optimal_weights_by_layer_number[layer_number][:, 0] = 0.0

    def eval_activations_by_feeding_forward(self):
        for layer_number in range(2, self.number_of_layers + 1):
            self.activation_values_by_layer_number[layer_number] = self.eval_activation_values(self.activation_values_including_ones_by_layer_number[layer_number - 1],
                                                                                               self.optimal_weights_by_layer_number[layer_number])
            if layer_number != self.number_of_layers:
                self.activation_values_including_ones_by_layer_number[layer_number] = self.append_bias_ones_to_array(self.activation_values_by_layer_number[layer_number])
            else:
                self.activation_values_including_ones_by_layer_number[layer_number] = self.activation_values_by_layer_number[layer_number]
            self.activation_derivatives_by_layer_number[layer_number] = self.eval_activation_derivatives(self.activation_values_including_ones_by_layer_number[layer_number])


    def eval_errors_by_back_propagation(self):
        for layer_number in reversed(range(2, self.number_of_layers)):
            self.error_by_layer_number[layer_number] = self.error_by_layer_number[layer_number + 1].\
                dot(self.optimal_weights_by_layer_number[layer_number + 1]) * self.activation_derivatives_by_layer_number[layer_number]

    def eval_cost_derivatives_vs_weights(self):
        for layer_number in range(2, self.number_of_layers + 1):
            self.derivatives_of_cost_vs_weights_by_layer[layer_number] = (self.activation_values_including_ones_by_layer_number[layer_number - 1].
                                                                          T.dot(self.error_by_layer_number[layer_number])).T
            if layer_number != self.number_of_layers:
                self.derivatives_of_cost_vs_weights_by_layer[layer_number] = self.derivatives_of_cost_vs_weights_by_layer[layer_number][1:, :]

    def eval_activation_values(self, activation_last_layer, weights_current_layer: array) -> array:
        '''
        :param weights_current_layer: shape of (number of nodes in current layer, number of nodes in the previous layer) 2d array
        :param activation_last_layer: shape of (number of datapoints, number of nodes in the previous layer)
        :param bias_current_layer: shape of (number of datapoints, number of nodes in the current layer)
        :return: a 2d array with shape (number of datapoints, number of nodes in the current layer) representing activation values at current layer
        '''
        return 1 / (1 + numpy.exp(-(activation_last_layer.dot(weights_current_layer.T))))

    def eval_activation_derivatives(self, activation_value: array) -> array:
        return activation_value * (1 - activation_value)

    def eval_output_layer_error(self, y_variables: array) -> array:
        """
        :return: shape of (number of datapoints, number of classes)
        """
        activation_value_output_layer = self.activation_values_by_layer_number[self.number_of_layers]
        activation_derivatives_output_layer = self.activation_derivatives_by_layer_number[self.number_of_layers]
        cost_vs_output_activation_derivatives = self.eval_derivative_of_cost_vs_output_activation(activation_value_output_layer, y_variables)
        result = cost_vs_output_activation_derivatives  * activation_derivatives_output_layer
        self.error_by_layer_number[self.number_of_layers] = result

    def convert_y_variables_to_binary(self, y_variables: array) -> array:
        unique_classes = numpy.unique(y_variables)
        # converting actual y data to equivalent probabilities
        y_variables_mask = numpy.array([y_variables] * self.number_of_output_classes).T
        class_mask = numpy.array([unique_classes] * y_variables.shape[0])
        y_variables_prob = (y_variables_mask == class_mask) * 1  # shape of (number of datapoints, number of classes)
        return y_variables_prob

    def eval_derivative_of_cost_vs_output_activation(self, output_layer_activation: array, y_variables: array):
        """
        :param output_layer_activation: shape of (number of datapoints, number of classes)
        :param y_variables: shape of (number of datapoints, ) 1d array
        :param cost_function: type of cost function, default to log
        :return: the derivative of cost function with respect to change in activation shape of (number of datapoints, number of classes)
        """
        # First need to convert the original y_variable to a shape of (number of datapoints, number of classes) with
        # probabilities of 0 or 1
        if self.cost_function == 'log':
            y_variables_prob = self.convert_y_variables_to_binary(y_variables)
            # Derivative of cost function against activation
            result = -y_variables_prob / output_layer_activation + (1 - y_variables_prob) / (1 - output_layer_activation)
            return result
        else:
            raise NotImplementedError('Cannot compute derivative for cost vs activation, reason: unknown cost function type.')

    def eval_cost_function(self, activation_values: array, y_variables: array):
        probability_of_each_sample = activation_values # shape (number of datapoints, number of classes)
        y_variables = self.convert_y_variables_to_binary(y_variables)
        a = numpy.log(1 - probability_of_each_sample)
        numpy.place(a, a == -numpy.inf, 0.0)
        b = numpy.log(probability_of_each_sample)
        numpy.place(b, b == -numpy.inf, 0.0)
        total_costs = -y_variables * b - (1 - y_variables) * a
        number_of_datapoints = len(total_costs)
        # if regularized_lambda is present, the weight of first feature is used as bias term.
        total_bias_term = 0
        for layer, weights in self.optimal_weights_by_layer_number.items():
            total_bias_term += numpy.sum(numpy.square(weights[:, 1:]))
        result = numpy.mean(total_costs) + self.regularized_lambda / (2 * number_of_datapoints) * total_bias_term
        return result

    def get_cost_weights_derivatives_by_approx(self, epsilon: float, weights: array, y_variables: array):
        upper_range = weights + epsilon # TODO this needs polish and checking
        lower_range = weights - epsilon
        derivatives = (self.eval_cost_function(upper_range, y_variables) -  self.eval_cost_function(lower_range, y_variables)) / (2 * epsilon)
        return derivatives

    def test_back_prop_gradient(self, epsilon: float):
        self.append_ones_to_activation(self.input_layer_x_variables_by_batch[1])
        self.eval_activations_by_feeding_forward()
        self.eval_output_layer_error(self.output_layer_y_variables_by_batch[1])
        self.eval_errors_by_back_propagation()
        self.eval_cost_derivatives_vs_weights()
        for layer_number in range(2, self.number_of_layers + 1):
            approx_gradient = self.get_cost_weights_derivatives_by_approx(epsilon, self.optimal_weights_by_layer_number[layer_number], self.output_layer_y_variables_by_batch[1])
            back_prop_gradient = self.derivatives_of_cost_vs_weights_by_layer[layer_number]
            no_of_inaccurate_gradients = numpy.sum((numpy.abs(approx_gradient - back_prop_gradient)) > 0.001)
            if no_of_inaccurate_gradients != 0:
                print(str(no_of_inaccurate_gradients))
                return False
        return True

    def regularized_weights(self, weights: array) -> array:
        weights = numpy.copy(weights)
        weights[:, 0] = 0
        weights[:, 1:] = self.regularized_lambda * weights[:, 1:]
        return weights

    def update_weights_and_errors_by_batch(self, batch_number: int):
        self.append_ones_to_activation(self.input_layer_x_variables_by_batch[batch_number])
        self.eval_activations_by_feeding_forward()
        self.eval_output_layer_error(self.output_layer_y_variables_by_batch[batch_number])
        self.eval_errors_by_back_propagation()
        self.eval_cost_derivatives_vs_weights()
        no_of_samples = self.input_layer_x_variables_by_batch[batch_number].shape[0]

        for layer_number in range(2, self.number_of_layers + 1):
            regularized_weights = self.regularized_weights(self.optimal_weights_by_layer_number[layer_number])
            self.optimal_weights_by_layer_number[layer_number] = self.optimal_weights_by_layer_number[layer_number] - \
                                                                 1 / no_of_samples * (self.learning_rate *
                                                                                      self.derivatives_of_cost_vs_weights_by_layer[layer_number] +
                                                                                      regularized_weights)

    def update_batches(self):
        for batch_number in self.input_layer_x_variables_by_batch:
            self.update_weights_and_errors_by_batch(batch_number)

    def train(self):
        for i in range(self.number_of_iteration):
            self.update_batches() # Getting optimal weights and bias
            # ================ Using the entire training set to evaluate output activation =====================
            self.append_ones_to_activation(self.training_x_variables)
            self.eval_activations_by_feeding_forward()
            # ==================================================================================================
            self.output_layer_training_activation = self.activation_values_by_layer_number[self.number_of_layers]
            self.training_y_results = self.output_layer_training_activation.argmax(axis=1) + 1
            accuracy = numpy.sum(self.training_y_variables == self.training_y_results) / len(self.training_y_variables)
            print('Epoch ' + str(i + 1) + ' training accuracy: ' + str(accuracy))

    def test(self):
        self.append_ones_to_activation(self.testing_x_variables)
        self.eval_activations_by_feeding_forward()
        self.output_layer_testing_activation = self.activation_values_by_layer_number[self.number_of_layers]
        self.testing_y_results = self.output_layer_testing_activation.argmax(axis=1) + 1
        accuracy = numpy.sum(self.testing_y_variables == self.testing_y_results) / len(self.testing_y_variables)
        print('Testing accuracy: ' + str(accuracy))

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
    input_data_dictionary['learning_rate'] = 1
    input_data_dictionary['no_of_training_batches'] = 4
    input_data_dictionary['layer_structure'] = {1: 400, 2: 25, 3: 10}
    nn = NeuralNetwork(**input_data_dictionary)
    nn.train()