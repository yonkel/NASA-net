import random
import time
import numpy as np
import matplotlib.pyplot as plt
from expnet_numpy import ExpNet
from net_util import Exp, Tahn, SigmoidNp
from generator import spirals, spiralsMinus
from perceptron_numpy import Perceptron

exp = Exp()
tahn = Tahn()
sigmoid = SigmoidNp()


def convergencia_spirals( architecture, net_type, act_func, learning_rate, max_epoch, repetitions, wanted_MSE , spiral_nodes, show ):
    # data_train, data_test, labels_train, labels_test
    inputs, test_inputs, labels, test_labels = spirals(spiral_nodes)

    threshold = 0.5
    lower_label = 0
    for item in labels:
        if item[0] == -1:
            threshold = 0
            lower_label = -1
            break
        if item[0] == 0:
            break


    start_time = time.time()
    nets_successful = 0
    epochs_to_success = []
    epoch_sum = 0
    p = len(inputs[0])
    # print(inputs[0])
    MSEs = []

    for n in range(repetitions):
        network = net_type(architecture, act_func, learning_rate)
        indexer = list(range(len(inputs)))
        epoch = 0

        MSE = []
        mse = 999
        while mse > wanted_MSE and epoch < max_epoch:
            random.shuffle(indexer)
            dobre = 0

            for i in indexer:
                intput = np.reshape(inputs[i], (2,1))
                act_hidden,act_output = network.activation(intput)
                network.learning(intput, act_hidden, act_output, labels[i])

                if act_output[0][0] >= threshold and labels[i][0] == 1 or act_output[0][0] < threshold and labels[i][0] == lower_label:
                    dobre += 1

            epoch += 1

            if epoch % 50 == 0:
                mse = network.MSE(test_inputs, test_labels)
                MSE.append(mse)

                print(f" Network {n}, epoch {epoch}, MSE {mse}, dobre {dobre} = {(dobre / len(inputs) * 100 )}%")



        if show:
            print("Spirals repetition {} sucess {}. Epochs to success: {}. MSE {} ".format(n,mse,epoch, mse ))
        epochs_to_success.append(epoch)

        if mse <=  wanted_MSE:
            nets_successful += 1
        epoch_sum += epoch

        MSEs.append(MSE)

    if show:
        print("\n{} networks out of {} converged to a solution".format(nets_successful,repetitions))
        plt.plot(list(range(repetitions)),epochs_to_success)
        plt.show()

    end_time = time.time()

    return {"nets": nets_successful, "epochs": epochs_to_success, "time": (end_time-start_time)}

if __name__ == '__main__':
    spiral_nodes = 1000
    architecture = [2, 15, 1]
    learning_rate = 0.9
    max_epoch = 3000
    repetitions = 10
    net_type = ExpNet
    act_fun = [tahn, tahn]
    wanted_MSE = 0.1

    x = convergencia_spirals( architecture, net_type, act_fun, learning_rate, max_epoch, repetitions, wanted_MSE , spiral_nodes , True )
    print(x)