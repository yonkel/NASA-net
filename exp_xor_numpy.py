import numpy as np
import matplotlib.pyplot as plt

from expnet_numpy import ExpNet
from net_util import SigmoidNp, Exp, Tahn



# inputs = np.array([[0,0],[0,1],[1,0],[1,1]])
inputs = np.array([[0, 0, 1, 1],[0, 1, 0, 1]]).astype(np.float32)
labels = np.array([[0, 1, 1, 0]]).astype(np.float32)

architecture = [2, 2, 1]
learning_rate = 0.5
max_epoch = 1000
minibatch_size = 1
dataset_size = len(inputs[0])

sigmoid = SigmoidNp()
exp = Exp()
tahn = Tahn()

network = ExpNet(architecture, [tahn,exp], learning_rate)
performance = []

for epoch in range(max_epoch):
    indexer = np.random.permutation(dataset_size)
    #print(indexer)
    success = 0
    for i in range(0, dataset_size, minibatch_size):

        input_batch = inputs[:, indexer[i:i+minibatch_size]]
        label_batch = labels[:, indexer[i:i+minibatch_size]]

        print(input_batch,  input_batch.shape)
        input("nieco")

        act_hidden,act_output = network.activation(input_batch)
        network.learning(input_batch, act_hidden, act_output, label_batch)
    _, test_output = network.activation(inputs)
    arg_out = (test_output > 0.5).astype(np.float32)
    score = np.sum(arg_out == labels)
    performance.append(score / dataset_size)
    print("Epoch {}. Expected: {} \t got: {} \t score: {}".format(epoch, labels, test_output, score))

plt.plot(list(range(max_epoch)),performance)
plt.show()
