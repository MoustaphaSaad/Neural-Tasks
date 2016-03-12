from __future__ import division
import numpy as np
import math

def step(val):
    if val < 0:
        return -1
    elif val > 0:
        return 1
    else:
        return 0

def lab2_update(neuron, error, eta, x):
    neuron.weights += (eta *error* x)
    return neuron

def lab3_update(neuron, error, eta, x = None):
    neuron.weights += (eta * error)
    return neuron

def lab2_error(y_, y):
    return y-y_

def LMS(x):
    return 0.5*x*x

class Perceptron:
    def __init__(self, act_func, d):
        self.activation_function = act_func
        self.weights = np.random.randn(1,d)
        self.bias = 1;

    def predict(self, x):
        v = np.sum(self.weights * x + self.bias)
        return self.activation_function(v)

    def getWeights(self):
        res = ""
        for w in self.weights[0]:
            res += str(w)
            res += ", "
        return res

class SingleNeuronNetwork:
    def __init__(self, neuron, error_func, update_func):
        self.neuron = neuron
        self.error_function = error_func
        self.update_function = update_func

    def train(self, train_data, test_data, eta = 0.1, epochs = 50):
        res = ""
        mse = []
        for i in xrange(epochs):
            error_cumulative  = 0
            errors = []
            for patch in train_data:
                for x, y in patch:
                    y_ = self.neuron.predict(x)
                    error = self.error_function(y_, y)
                    errors.append(error)
                    error_cumulative += error * x
            self.neuron = self.update_function(self.neuron, error_cumulative, eta, None)
            mse.append(np.mean(LMS(np.array(errors))))
            res += "Epoch#" + str(i) + ": "
            res += self.neuron.getWeights()
            res += "\n"
        return res

    def test(self, test_data):
        conf_mat = [[0 for i in xrange(3)]for i in xrange(3)]
        total_error = 0
        for x, y in test_data:
            y_ = self.neuron.predict(x)
            ixy = 0
            ixy_ = 0
            if y_ == -1:
                ixy_ = 0
            elif y_ == 1:
                ixy_ = 1

            if y == -1:
                ixy = 0
            elif y == 1:
                ixy = 1

            conf_mat[ixy_][ixy] += 1

            error = self.error_function(y_,y)
            if y_ != y:
                total_error += 1

        return conf_mat, total_error
