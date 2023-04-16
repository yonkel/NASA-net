from net_util import Tahn, Quasi, SigmoidNp
from generator import paritaMinus, parita, banana, spiralsMinusTransformed, spiralsMinus
import random
import time
import numpy as np
import matplotlib.pyplot as plt
from NASA import NASA
from util import get_threshold
from sklearn.model_selection import KFold


def convergence_spirals(net_type, net_hyperparams, repetitions, max_epoch, success_window, number_of_points, show=False,
                        plt_name=""):
    inputs, labels = spiralsMinusTransformed(number_of_points)
    threshold, label = get_threshold(labels)

    nets_successful = 0
    epochs_to_success = []
    MSE_all = []
    good_outputs_all = []

    indexer = list(range(len(inputs)))
    x_dim = len(inputs[0])

    start_time = time.time()

    for n in range(repetitions):

        network = net_type(net_hyperparams)

        epoch = 0
        max_good_outputs = 0
        window = 0

        MSE_tracker = []
        good_outputs_tracker = []

        while epoch < max_epoch:
            MSE = 0
            good_outputs = 0
            for i in indexer:
                x = np.reshape(inputs[i], (x_dim, 1))
                h = network.activation(x)
                y = h[-1]
                network.learning(h, labels[i])

                MSE += (y - labels[i][0]) ** 2

            if y[0][0] >= threshold and labels[i][0] == 1 or y[0][0] < threshold and labels[i][0] == label:
                # print(y[0][0], labels[i][0], threshold)
                good_outputs += 1

            if good_outputs > max_good_outputs:
                max_good_outputs = good_outputs

            MSE = np.squeeze(MSE / len(inputs))

            MSE_tracker.append(MSE)
            good_outputs_tracker.append(good_outputs)

            if good_outputs == len(inputs):
                window += 1
            else:
                window = 0

            if window == success_window:
                nets_successful += 1
                break

            print("repetiton", n, f": {good_outputs}/{len(inputs)}, MSE = {MSE}")
            epoch += 1

        if show:
            print(
                f"Parity repetition {n} converged {(window == success_window)}. Epochs reached: "
                f"{epoch}. {max_good_outputs} out of {len(inputs)}.")

        epochs_to_success.append((n, epoch, "blue" if window == success_window else "red"))
        MSE_all.append(MSE_tracker)
        good_outputs_all.append(good_outputs_tracker)

    end_time = time.time()

    if show:
        print(f"\n{nets_successful} networks out of {repetitions} converged to a solution. {end_time - start_time}")

        for i in range(repetitions):
            plt.plot(MSE_all[i], label=f"repetition {i}")

        plt.show()

    return {"nets": nets_successful, "epochs": [e[1] for e in epochs_to_success], "time": (end_time - start_time),
            "MSE": MSE_all, "good_outputs": good_outputs_all}


def k_fold_spirals(net_type, net_hyperparams, repetitions, max_epoch, number_of_points, show=False,
                   plt_name=""):
    inputs, labels = spiralsMinusTransformed(number_of_points)
    threshold, label = get_threshold(labels)

    MSE_all = []
    good_outputs_all = []

    x_dim = len(inputs[0])

    start_time = time.time()

    kf = KFold(5, shuffle=True)
    for train_index, test_index in kf.split(inputs, labels):
        for n in range(repetitions):
            network = net_type(net_hyperparams)

            epoch = 0

            while epoch < max_epoch:

                np.random.shuffle(train_index)

                good_outputs_train = 0
                MSE_train = 0
                for i in train_index:
                    x = np.reshape(inputs[i], (x_dim, 1))
                    h = network.activation(x)
                    y = h[-1]
                    network.learning(h, labels[i])

                    if y[0][0] >= threshold and labels[i][0] == 1 or y[0][0] < threshold and labels[i][0] == label:
                        good_outputs_train += 1

                    MSE_train += (labels[i][0] - y) ** 2

                MSE_train /= len(train_index)

                print(
                    f"repetition {n}, epoch {epoch}, good outputs {good_outputs_train} / {len(train_index)}, MSE {MSE_train}")

                epoch += 1

            good_outputs_test = 0
            MSE_test = 0
            for i in test_index:
                x = np.reshape(inputs[i], (x_dim, 1))
                h = network.activation(x)
                y = h[-1]

                if y[0][0] >= threshold and labels[i][0] == 1 or y[0][0] < threshold and labels[i][0] == label:
                    good_outputs_test += 1

                MSE_test += (labels[i][0] - y) ** 2

            MSE_test = np.squeeze(MSE_test / len(inputs))

            good_outputs_all.append(good_outputs_test)
            MSE_all.append(MSE_test)

            if show:
                print(
                    f"Repetition {n}, {epoch}\n"
                    f"test good_outputs : {good_outputs_test} / {len(test_index)}, {MSE_test}\n"
                    f"train good_outputs : {good_outputs_train} / {len(train_index)}, {MSE_train}\n ")

    end_time = time.time()

    if show:
        print(f"\ngood_outputs_all:{good_outputs_all} \n  {end_time - start_time}")

        plt.scatter(MSE_all, list(range(len(MSE_all))), color="blue")
        plt.xlabel("repetition")
        plt.ylabel("MSE")
        plt.show()

        plt.scatter(good_outputs_all, list(range(len(good_outputs_all))), color="red")
        plt.xlabel("repetition")
        plt.ylabel("good outputs")
        plt.show()

    return {"MSE": MSE_all, "good_outputs": good_outputs_all, "max_good_test": len(test_index)}


if __name__ == "__main__":
    tahn = Tahn()
    quasi = Quasi()
    hyper_params = {
        "layers": [2, 10, 10, 1],
        "activation_functions": [quasi, quasi, tahn],
        "learning_rate": 0.005,
        "weight_mean": 0.0,
        "weight_variance": 1
    }

    # net_type, net_hyperparams, repetitions, max_epoch,number_of_points, show=False,
    print(k_fold_spirals(NASA, hyper_params, 2, 500, 200, True))
