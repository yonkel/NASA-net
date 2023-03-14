from net_util import Tahn, Quasi, SigmoidNp
from generator import paritaMinus, parita, banana, spiralsMinusTransformed, spiralsMinus
import random
import time
import numpy as np
import matplotlib.pyplot as plt
from NASA import NASA
from util import get_threshold
from sklearn.model_selection import cross_val_score


def convergence_mse(net_type, net_hyperparams, repetitions, max_epoch, success_window, number_of_points, show=False,
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


if __name__ == "__main__":
    p = 2
    inputs, labels = paritaMinus(p)

    # set network PARAMS
    sigm = SigmoidNp()
    tahn = Tahn()
    quasi = Quasi()
    hyper_params = {
        "layers": [2, 15, 15, 1],
        "activation_functions": [quasi, quasi, tahn],
        "learning_rate": 0.005,
        "weight_mean": 0.0,
        "weight_variance": 1
    }

    # print(convergence(NASA, hyper_params, 5, 1000, 10, inputs, labels, True))
    print(convergence_mse(NASA, hyper_params, 2, 50, 10, 200, True))
