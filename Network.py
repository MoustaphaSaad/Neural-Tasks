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

def lab2_error(y_, y):
    return y-y_

class Perceptron:
    def __init__(self, act_func, d):
        self.activation_function = act_func
        self.weights = np.random.randn(1,d)
        self.bias = 1;

    def predict(self, x):
        v = np.sum(self.weights * x + self.bias)
        return self.activation_function(v)

class SingleNeuronNetwork:
    def __init__(self, neuron, error_func, update_func):
        self.neuron = neuron
        self.error_function = error_func
        self.update_function = update_func

    def train(self, train_data, test_data, eta = 0.1, epochs = 50):

        for i in xrange(epochs):
            for x, y in train_data:

                y_ = self.neuron.predict(x)

                error = self.error_function(y_,y)
                self.neuron = self.update_function(self.neuron, error, eta, x)
            print self.neuron.weights
            self.test(test_data)

    def test(self, test_data):
        total_error = 0
        for x, y in test_data:
            y_ = self.neuron.predict(x)

            error = self.error_function(y_,y)
            total_error += error

        print total_error
