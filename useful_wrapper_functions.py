import pandas
import numpy
from sklearn import linear_model


def fit_GDregression(x, y, typ=None):
    '''
    Gradient descent algo that returns a numpy array (predicted Y)
    '''
    if typ is None:
        regression = linear_model.SGDRegressor(loss='squared_loss')
        regression.fit(x,y)
        coef_array = numpy.hstack((regression.intercept_, regression.coef_))
        linear_func = numpy.poly1d(coef_array[::-1])
        print(numpy.poly1d(linear_func))
        print('R squre is: ' + str(regression.score(x,y)))
        return regression.predict(x)
    elif typ == 'logistic':
        regression = linear_model.SGDClassifier(loss='log')
        regression.fit(x,y)
        print('accuracy is: ' + str(regression.score(x,y)))
        return regression
    
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