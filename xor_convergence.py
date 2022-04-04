import random
import matplotlib.pyplot as plt
from expnet_numpy import ExpNet
from net_util import SigmoidNp, Exp, Tahn
exp = Exp()
tahn = Tahn()
import numpy as np
from generator import parita

p = 4
inputs, labels = parita(p)
print(len(inputs))

# inputs = [[0,0],[0,1],[1,0],[1,1]]
# labels = [[0], [1], [1], [0]]
architecture = [p,40,1]
learning_rate = 0.5
max_epoch = 5000

repetitions = 100
success_window = 10
epochs_to_success = []
nets_successful = 0

for n in range(repetitions):
    network = ExpNet(architecture, [tahn, exp], learning_rate)
    indexer = list(range(len(inputs)))
    success_global = 0
    epoch = 0
    while success_global < success_window and epoch < max_epoch:
        random.shuffle(indexer)
        success_epoch = 0
        for i in indexer:
            input = np.reshape(np.array(inputs[i]), (p, 1))
            act_hidden,act_output = network.activation(input)
            # print("e{0}: {1} >> {2} | {3}".format(epoch+1, inputs[i], act_output, labels[i]))
            if act_output[0] >= 0.5 and labels[i][0] == 1 or act_output[0] < 0.5 and labels[i][0] == 0:
                success_epoch += 1
            network.learning(input, act_hidden, act_output, labels[i])
        if success_epoch == 2**p:
            success_global += 1
        epoch += 1
    print("XOR repetition {} sucess {}. Epochs to success: {}. Success : {}".format(n,(success_epoch == 2**p),epoch, success_epoch))
    epochs_to_success.append(epoch)
    if success_epoch == 2**p:
        nets_successful += 1

print("\n{} networks out of {} converged to a solution".format(nets_successful,repetitions))

plt.plot(list(range(repetitions)),epochs_to_success)
plt.show()