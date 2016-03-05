from __future__ import division
import Loader
import Network
import numpy as np

loader = Loader.IrisLoader("Data/Iris Data.txt", "r")
result = loader.load()

data = []

filter_X1 = 1
filter_X2 = 2

for Class in result:
    for i in xrange(len(result[Class])):
        sample = result[Class][i]
        result[Class][i] = np.array([sample[filter_X1], sample[filter_X2]])

#("Iris-setosa", -1)
#("Iris-virginica", -1)
#("Iris-versicolor", 1)
Classes = [("Iris-setosa", -1),  ("Iris-versicolor", 1)]


for Class, label in Classes:
    iter_label = np.empty(len(result[Class]))
    iter_label.fill(label)
    dd = zip(result[Class],iter_label)
    data.append(dd)

train_data = []
test_data = []

for Class in data:
    train_data.extend(Class[0:30])
    test_data.extend(Class[30:50])

net = Network.SingleNeuronNetwork(Network.Perceptron(Network.step,2),Network.lab2_error,Network.lab2_update)
net.train(train_data,test_data,0.3,50)

net.test(test_data)
