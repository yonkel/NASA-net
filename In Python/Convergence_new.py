from net_util import Tahn, Quasi, SigmoidNp
from generator import paritaMinus, parita
import random
import time
import numpy as np
import matplotlib.pyplot as plt
from NASA import NASA

def convergence(net_type, net_hyperparams, repetitions, max_epoch, success_window, inputs, labels, show=False, plt_name = ""):

    threshold = 0.5
    label = 0
    for item in labels:
        if item[0] == -1:
            threshold = 0
            label = -1
            break
        if item[0] == 0:
            break

    nets_successful = 0
    epochs_to_success = []

    p = len(inputs[0])

    start_time = time.time()
    for n in range(repetitions):
        network = net_type(net_hyperparams)

        epoch = 0
        max_good_outputs = 0
        window = 0

        indexer = list(range(len(inputs)))
        while epoch < max_epoch:
            random.shuffle(indexer)
            good_outputs = 0
            for i in indexer:

                x = np.reshape(inputs[i], (p, 1))
                h = network.activation(x)
                y = h[-1]

                if y[0][0] >= threshold and labels[i][0] == 1 or y[0][0] < threshold and labels[i][0] == label:
                    good_outputs += 1

                network.learning(h, labels[i])

                if good_outputs > max_good_outputs:
                    max_good_outputs = good_outputs

            if good_outputs == 2 ** p:
                window += 1
            else:
                window = 0

            if window == success_window:
                nets_successful += 1
                break

            epoch += 1

        if show:
            print(
                f"Parity repetition {n} converged {(window == success_window)}. Epochs reached: {epoch}. {max_good_outputs} out of {2 ** p}.")

        epochs_to_success.append((n,epoch,"blue" if window == success_window else "red"))

    end_time = time.time()

    if show:
        print(f"\n{nets_successful} networks out of {repetitions} converged to a solution. {end_time - start_time}")
        plt.scatter([i[0] for i in epochs_to_success], [i[1] for i in epochs_to_success], c=[i[2] for i in epochs_to_success])

        act_title = ""
        for fun in net_hyperparams["activation_functions"]:
            if type(fun) == Quasi:
                act_title += "Quasi "
            elif type(fun) == Tahn:
                act_title += "Tahn "
            elif type(fun) == SigmoidNp:
                act_title += "Sigmo "
            else:
                print("sprav to poriadne potom ")

        plt.title(f'{act_title} {net_hyperparams["layers"]}')
        plt.show()

    return {"nets": nets_successful, "epochs": [ e[1] for e in epochs_to_success], "time": (end_time - start_time)}



if __name__ == "__main__":
    p = 7
    inputs, labels = parita(p)


    # set network PARAMS
    sigm = SigmoidNp()
    tahn = Tahn()
    quasi = Quasi()
    hyper_params = {
        "layers": [p, 100, 100, 1],
        "activation_functions": [ sigm, sigm, sigm],
        "learning_rate": 0.5,
        "weight_mean": 0.0,
        "weight_variance": 1
    }


    convergence(NASA, hyper_params, 20, 5000, 5, inputs, labels, True)



# # DODO ide ?
# network = NASA(params)
# success_global = 0
# epoch = 0
# succ_max = 0
# success_window = 10
# max_epoch = 1000
# indexer = list(range(len(inputs)))
#
# while success_global < success_window and epoch < max_epoch:
#     # random.shuffle(indexer)
#     success_epoch = 0
#     for i in indexer:
#         intput = np.reshape(inputs[i], (p, 1))
#         h = network.activation(intput)
#         y = h[-1]
#
#         if y[0][0] >= threshold and labels[i][0] == 1 or y[0][0] < threshold and labels[i][0] == label:
#             success_epoch += 1
#         network.learning(h, labels[i])
#
#         if success_epoch > succ_max:
#             succ_max = success_epoch
#
#     if success_epoch == 2 ** p:
#         success_global += 1
#
#     epoch += 1
#
# print(f"XOR sucess {(success_epoch == 2 ** p)}. Epochs to success: {epoch}. {succ_max} out of {2 ** p}")
