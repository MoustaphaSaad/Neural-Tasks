from __future__ import division
import Loader
import Network
import numpy as np
from socket import *
import time

HOST = ''
PORT = 29876
ADDR = (HOST, PORT)
BUFSIZE = 4096
serv = socket(AF_INET, SOCK_STREAM)

serv.bind((ADDR))
serv.listen(5)

conn, addr = serv.accept()

conn.send('TEST_HANDSHAKE')

filter_X1 = 1
filter_X2 = 2
filter_C1 = 0
filter_C2 = 1

filter_C1 = int(conn.recv(BUFSIZE))
filter_C2 = int(conn.recv(BUFSIZE))

filter_X1 = int(conn.recv(BUFSIZE))
filter_X2 = int(conn.recv(BUFSIZE))

ETA = 0.1
ETA = float(conn.recv(BUFSIZE))

EPOCHS = 50
EPOCHS = int(conn.recv(BUFSIZE))


All_CLASSES = ["Iris-setosa",  "Iris-versicolor", "Iris-virginica"]

DATA_LOC = "G:/Hell/Neural/Tasks/Data/Iris Data.txt"
DATA_LOC = conn.recv(BUFSIZE)
loader = Loader.IrisLoader(DATA_LOC, "r")
result = loader.load()

data = []

for Class in result:
    for i in xrange(len(result[Class])):
        sample = result[Class][i]
        result[Class][i] = np.array([sample[filter_X1], sample[filter_X2]])

#("Iris-setosa", -1)
#("Iris-virginica", -1)
#("Iris-versicolor", 1)
Classes = [(All_CLASSES[filter_C2], -1),  (All_CLASSES[filter_C1], 1)]
print Classes

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
log = net.train(train_data,test_data,ETA,EPOCHS)

conf_mat, total_error = net.test(test_data)
conf_mat_str = ""
for i in xrange(len(conf_mat)):
    for j in xrange(len(conf_mat[i])):
        conf_mat_str += str(conf_mat[i][j])
        conf_mat_str += ", "

conn.send(conf_mat_str)
gw = net.neuron.getWeights()
conn.send(gw)
time.sleep(1)
conn.send(str(total_error))
time.sleep(1)
conn.send(log)
conn.close()
