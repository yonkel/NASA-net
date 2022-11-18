import random
import time
import numpy as np
import matplotlib.pyplot as plt
from NASA import ExpNet
from net_util import Exp, Tahn, SigmoidNp
from generator import spirals, spiralsMinus, spiralsMinusTransformed
from perceptron import Perceptron
from statistics import mean, stdev
from util import check_dir

exp = Exp()
tahn = Tahn()
sigmoid = SigmoidNp()


def save( exp_name, net_name,  value, what, epochs, _mean, _stdev):
    lines =  [ f"{epochs[i]} {_mean[i]} {_stdev[i]}\n" for i in range(len(epochs))  ]

    check_dir("results")

    with open(f"results/{net_name}_{exp_name}_{value}_{what}.txt", 'a') as file:
        file.write("x y err\n")
        file.writelines(lines)

def convergence_general( architecture, net_type, act_func, learning_rate, max_epoch, repetitions, wanted_MSE , data, show, save_params ):
    # data_train, data_test, labels_train, labels_test
    inputs, test_inputs, labels, test_labels = data

    start_time = time.time()
    nets_successful = 0
    epochs_to_success = []
    epoch_sum = 0
    p = len(inputs[0])
    # print(inputs[0])

    ACC_test_all = []
    ACC_train_all = []
    MSE_all = []

    for n in range(repetitions):
        network = net_type(architecture, act_func, learning_rate)
        indexer = list(range(len(inputs)))
        epoch = 0


        MSE_repetition = []
        ACC_train_repetition = []
        ACC_test_repetition = []

        epochs = []
        epoch_flag = True

        while epoch < max_epoch :
            random.shuffle(indexer)
            SSE = 0
            properly_determined_train = 0

            for i in indexer:
                intput = np.reshape(inputs[i], (2,1))
                act_hidden, act_output = network.activation(intput)
                network.learning(intput, act_hidden, act_output, labels[i])
                SSE += (labels[i][0] - act_output[0]) ** 2


            MSE = SSE / len(labels)
            epoch += 1
            properly_determined_train = network.properly_determined(inputs, labels)
            properly_determined_test = network.properly_determined(test_inputs, test_labels)


            if epoch % 10 == 0:
                print(f" Network {n}, epoch {epoch}, MSE {MSE}, ACC_train {properly_determined_train} = {round((properly_determined_train / len(inputs) * 100), 2 ), }%, ACC_test {properly_determined_test} = {round((properly_determined_test / len(test_labels) * 100), 2 ), }%")


            MSE_repetition.append(MSE[0])
            ACC_test_repetition.append(properly_determined_test / len(test_labels))
            ACC_train_repetition.append(properly_determined_train / len(labels))

            if epoch_flag:
                epochs.append(epoch)

        # if show:
        #     print("Convergence repetition {} Epochs to success: {}. MSE {}. ACC {} = {}% ".format(n,epoch, MSE[0], properly_determined, round((properly_determined / len(inputs) * 100), 2 ) ))

        epochs_to_success.append(epoch)
        epoch_sum += epoch

        ACC_test_all.append(ACC_test_repetition)
        ACC_train_all.append(ACC_train_repetition)
        MSE_all.append(MSE_repetition)

        if MSE_repetition[-1] <= wanted_MSE:
            nets_successful += 1

        epoch_flag = False

    if show:
        print("\n{} networks out of {} converged to a solution".format(nets_successful,repetitions))
        plt.plot(list(range(repetitions)),epochs_to_success)
        # plt.show()
        plt.savefig("spirals_exp_{}.pdf".format(time.time()), format="pdf")

    end_time = time.time()

    ACC_test_mean = []
    ACC_train_mean = []
    MSE_mean = []

    ACC_test_stdev = []
    ACC_train_stdev = []
    MSE_stdev = []

    for i in range(len(MSE_all[0])):
        epoch_test_ACC = [el[i] for el in ACC_test_all]
        epoch_train_ACC = [el[i] for el in ACC_train_all]
        epoch_MSE = [el[i] for el in MSE_all]

        ACC_test_mean.append(mean(epoch_test_ACC))
        ACC_train_mean.append(mean(epoch_train_ACC))
        MSE_mean.append(mean(epoch_MSE))

        ACC_test_stdev.append(stdev(epoch_test_ACC))
        ACC_train_stdev.append(stdev(epoch_train_ACC))
        MSE_stdev.append(stdev(epoch_MSE))

    # all_epochs = range(0, max_epoch, max_epoch // len(ACC_all[0]))

    save(save_params["exp_name"], save_params["net_name"], save_params["value"], "ACC_test", epochs, ACC_test_mean, ACC_test_stdev)
    save(save_params["exp_name"], save_params["net_name"], save_params["value"], "ACC_train", epochs, ACC_train_mean, ACC_train_stdev)
    save(save_params["exp_name"], save_params["net_name"], save_params["value"], "MSE", epochs, MSE_mean, MSE_stdev )

    return {"nets": nets_successful, "epochs": epochs_to_success, "time": (end_time-start_time) }

if __name__ == '__main__':
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
