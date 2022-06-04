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


def convergence_general( architecture, net_type, act_func, learning_rate, max_epoch, repetitions, wanted_MSE , data, show ):
    # data_train, data_test, labels_train, labels_test
    inputs, test_inputs, labels, test_labels = data

    properly_determined_all = []
    start_time = time.time()
    nets_successful = 0
    epochs_to_success = []
    epoch_sum = 0
    p = len(inputs[0])
    # print(inputs[0])
    MSE = 0

    for n in range(repetitions):
        network = net_type(architecture, act_func, learning_rate)
        indexer = list(range(len(inputs)))
        epoch = 0

        properly_determined = 0

        mse = 999
        while epoch < max_epoch :
            random.shuffle(indexer)
            properly_determined = 0

            for i in indexer:
                intput = np.reshape(inputs[i], (2,1))
                act_hidden, act_output = network.activation(intput)
                network.learning(intput, act_hidden, act_output, labels[i])



            epoch += 1


            if epoch % 500 == 0:
                mse, properly_determined = network.MSE(test_inputs, test_labels)
                print(f" Network {n}, epoch {epoch}, MSE {mse}, properly determined {properly_determined} = {round((properly_determined / len(inputs) * 100), 2 )}%")




        if show:
            print("Convergence repetition {} sucess {}. Epochs to success: {}. MSE {} ".format(n,mse,epoch, mse ))
        epochs_to_success.append(epoch)

        if mse <= wanted_MSE or properly_determined / len(inputs) >= 0.95:
            nets_successful += 1
        epoch_sum += epoch

        MSE += mse
        properly_determined_all.append(properly_determined)

    if show:
        print("\n{} networks out of {} converged to a solution".format(nets_successful,repetitions))
        plt.plot(list(range(repetitions)),epochs_to_success)
        plt.show()

    end_time = time.time()

    return {"nets": nets_successful, "epochs": epochs_to_success, "time": (end_time-start_time), "mse": MSE/repetitions, "properly_determined" : properly_determined_all }

if __name__ == '__main__':
    spiral_nodes = 1000
    architecture = [2, 80, 1]
    learning_rate = 0.9
    max_epoch = 5000
    repetitions = 10
    net_type = ExpNet
    act_fun = [tahn, tahn]
    wanted_MSE = 0.1
    data = spirals(500)

    x = convergence_general( architecture, net_type, act_fun, learning_rate, max_epoch, repetitions, wanted_MSE , data , True )
    print(x)