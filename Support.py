import pandas
import numpy
from sklearn import linear_model
import matplotlib.pyplot as plt

def get_data(path):
    data = pandas.read_csv(path, header=None)
    return data

def fit_GDregression(x, y, n_iter=100, alpha=0.0001):
    '''
    Gradient descent algo that returns a numpy array (predicted Y)
    '''
    regression = linear_model.SGDRegressor(n_iter=n_iter, alpha=alpha)
    regression.fit(x,y)
    coef_array = numpy.hstack((regression.intercept_, regression.coef_))
    linear_func = numpy.poly1d(coef_array[::-1])
    print numpy.poly1d(linear_func)
    print 'R squre is: ' + str(regression.score(x,y))
    return regression.predict(x)

def normal_equation(x, y):
    '''
    Normal equation to compute optimal coefficients
    return a numpy array
    '''
    one_added_array = numpy.hstack((numpy.ones((len(x.index),1)),x.values))
    x_matrix = numpy.matrix(one_added_array)
    y_matrix = (numpy.matrix(y)).getT()
    x_transpose_matrix = x_matrix.getT()
    result = (((x_transpose_matrix.dot(x_matrix)).I).dot(x_transpose_matrix)).dot(y_matrix)
    return result.A