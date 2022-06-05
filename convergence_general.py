import random
import time
import numpy as np
import matplotlib.pyplot as plt
from expnet_numpy import ExpNet
from net_util import Exp, Tahn, SigmoidNp
from generator import spirals, spiralsMinus, spiralsMinusTransformed
from perceptron_numpy import Perceptron
from statistics import mean, stdev

exp = Exp()
tahn = Tahn()
sigmoid = SigmoidNp()


def convergence_general( architecture, net_type, act_func, learning_rate, max_epoch, repetitions, wanted_MSE , data, show ):
    # data_train, data_test, labels_train, labels_test
    inputs, test_inputs, labels, test_labels = data


    start_time = time.time()
    nets_successful = 0
    epochs_to_success = []
    epoch_sum = 0
    p = len(inputs[0])
    # print(inputs[0])

    ACC_all = []
    MSE_all = []

    for n in range(repetitions):
        network = net_type(architecture, act_func, learning_rate)
        indexer = list(range(len(inputs)))
        epoch = 0

        properly_determined = 0

        MSE_repetition = []
        ACC_repetition = []

        while epoch < max_epoch :
            random.shuffle(indexer)
            SSE = 0
            for i in indexer:
                intput = np.reshape(inputs[i], (2,1))
                act_hidden, act_output = network.activation(intput)
                network.learning(intput, act_hidden, act_output, labels[i])
                SSE += (labels[i][0] - act_output[0]) ** 2


            MSE = SSE / len(labels)
            epoch += 1
            properly_determined = network.properly_determined(test_inputs, test_labels)


            if epoch % 10 == 0:
                print(f" Network {n}, epoch {epoch}, MSE {MSE}, ACC {properly_determined} = {round((properly_determined / len(inputs) * 100), 2 )}%")

            MSE_repetition.append(MSE[0])
            ACC_repetition.append(round((properly_determined / len(inputs) * 100), 2 ))


        if show:
            print("Convergence repetition {} Epochs to success: {}. MSE {}. ACC {} = {}% ".format(n,epoch, MSE[0], properly_determined, round((properly_determined / len(inputs) * 100), 2 ) ))
        epochs_to_success.append(epoch)


        epoch_sum += epoch

        MSE_all.append(MSE_repetition)
        ACC_all.append(ACC_repetition)

        if MSE_repetition[-1] <= wanted_MSE:
            nets_successful += 1

    if show:
        print("\n{} networks out of {} converged to a solution".format(nets_successful,repetitions))
        plt.plot(list(range(repetitions)),epochs_to_success)
        # plt.show()
        plt.savefig("spirals_exp_{}.pdf".format(time.time()), format="pdf")

    end_time = time.time()

    ACC_mean = []
    MSE_mean = []

    ACC_stdev = []
    MSE_stdev = []

    for i in range(len(ACC_mean[0])):
        epoch_ACC = [el[0] for el in ACC_all]
        epoch_MSE = [el[0] for el in MSE_all]

        ACC_mean.append(mean(epoch_ACC))
        MSE_mean.append(mean(epoch_MSE))

        ACC_stdev.append(stdev(epoch_ACC))
        MSE_stdev.append(stdev(epoch_MSE))

    return {"nets": nets_successful, "epochs": epochs_to_success, "time": (end_time-start_time), "mse_mean": MSE_mean, "acc_mean" : ACC_mean }

if __name__ == '__main__':
    spiral_nodes = 1000
    architecture = [2, 80, 1]
    learning_rate = 0.9
    max_epoch = 5000
    repetitions = 10
    net_type = ExpNet
    act_fun = [tahn, tahn]
    wanted_MSE = 0.1
    data = spiralsMinusTransformed(500)

    x = convergence_general( architecture, net_type, act_fun, learning_rate, max_epoch, repetitions, wanted_MSE , data , True )
    print(x)
