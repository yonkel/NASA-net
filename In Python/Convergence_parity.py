from net_util import Tahn, Quasi, SigmoidNp
from generator import paritaMinus, parita, banana, spiralsMinusTransformed, spiralsMinus
import random
import time
import numpy as np
import matplotlib.pyplot as plt
from NASA import NASA
from util import get_threshold


def convergence_parity(net_type, net_hyperparams, repetitions, max_epoch, success_window, inputs, labels, show=False,
                plt_name=""):

    threshold, label = get_threshold(labels)

    nets_successful = 0
    epochs_to_success = []

    x_dim = len(inputs[0])

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

                x = np.reshape(inputs[i], (x_dim, 1))
                h = network.activation(x)
                y = h[-1]

                if y[0][0] >= threshold and labels[i][0] == 1 or y[0][0] < threshold and labels[i][0] == label:
                    # print(y[0][0], labels[i][0], threshold)
                    good_outputs += 1

                network.learning(h, labels[i])

                if good_outputs > max_good_outputs:
                    max_good_outputs = good_outputs

            if good_outputs == len(inputs):
                window += 1
            else:
                window = 0

            if window == success_window:
                nets_successful += 1
                break

            print("repetiton", n, f"{good_outputs}/{len(inputs)}")
            epoch += 1

        if show:
            print(
                f"Parity repetition {n} converged {(window == success_window)}. Epochs reached: {epoch}. {max_good_outputs} out of {len(inputs)}.")

        epochs_to_success.append((n, epoch, "blue" if window == success_window else "red"))

    end_time = time.time()

    if show:
        print(f"\n{nets_successful} networks out of {repetitions} converged to a solution. {end_time - start_time}")
        plt.scatter([i[0] for i in epochs_to_success], [i[1] for i in epochs_to_success],
                    c=[i[2] for i in epochs_to_success])

        act_title = ""
        for fun in net_hyperparams["activation_functions"]:
            act_title += str(fun)
            act_title += " "

        plt.title(f'{act_title} {net_hyperparams["layers"]}')
        plt.show()

    return {"nets": nets_successful, "epochs": [e[1] for e in epochs_to_success], "time": (end_time - start_time)}



#cross entropy


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

    data = spiralsMinusTransformed(200)

    # print(convergence(NASA, hyper_params, 5, 1000, 10, inputs, labels, True))
    print(convergence_parity(NASA, hyper_params, 2, 50, 10, data[0].tolist(), data[2].tolist(), True))

