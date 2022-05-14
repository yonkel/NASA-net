import random
import time
import numpy as np
import matplotlib.pyplot as plt
from expnet_numpy import ExpNet
from net_util import Exp, Tahn
from generator import paritaMinus
exp = Exp()
tahn = Tahn()


def convergencia( architecture, net_type, act_func, learning_rate, max_epoch, repetitions, success_window, inputs, labels, show ):
    start_time = time.time()
    nets_successful = 0
    epochs_to_success = []
    epoch_sum = 0
    p = len(inputs[0])
    # print(inputs[0])
    for n in range(repetitions):
        network = net_type(architecture, act_func, learning_rate)
        indexer = list(range(len(inputs)))
        success_global = 0
        epoch = 0
        succ_max = 0
        while success_global < success_window and epoch < max_epoch:
            random.shuffle(indexer)
            success_epoch = 0
            for i in indexer:
                intput = np.reshape(inputs[i], (p,1))
                act_hidden,act_output = network.activation(intput)
                # print("e{0}: {1} >> {2} | {3}".format(epoch+1, inputs[i], act_output, labels[i]))
                if act_output[0][0] >= 0.5 and labels[i][0] == 1 or act_output[0][0] <= -0.5 and labels[i][0] == -1:
                    success_epoch += 1
                network.learning(intput, act_hidden, act_output, labels[i])
            if success_epoch == 2**p:
                success_global += 1
            if success_epoch > succ_max:
                succ_max = success_epoch
            epoch += 1
        if show:
            print("XOR repetition {} sucess {}. Epochs to success: {}. {} out of {}".format(n,(success_epoch == 2**p),epoch, succ_max, 2**p ))
        epochs_to_success.append(epoch)
        if success_global == success_window:
            nets_successful += 1
        epoch_sum += epoch
        # if you want avg only from successful nets, tab the line above

    if show:
        print("\n{} networks out of {} converged to a solution".format(nets_successful,repetitions))
        plt.plot(list(range(repetitions)),epochs_to_success)
        plt.show()

    end_time = time.time()

    return {"nets": nets_successful, "epochs": epochs_to_success, "time": (end_time-start_time)}


if __name__ == "__main__":
    pass
    p = 3
    inputs, labels = paritaMinus(p)

    architecture = [p,3,1]
    learning_rate = 0.5
    max_epoch = 1000
    repetitions = 100
    success_window = 10
    net_type = ExpNet
    act_fun = [tahn, exp]

    x = convergencia( architecture, net_type, act_fun, learning_rate, max_epoch, repetitions, success_window, inputs, labels, True)
    print(x)