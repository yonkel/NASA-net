import time
from NASA import NASA
from net_util import Quasi, Tahn, SigmoidNp
from generator import paritaMinus
from Convergence_new import convergence
from statistics import mean, stdev
from util import save_results

p = 7
inputs, labels = paritaMinus(p)

# set network PARAMS
sigm = SigmoidNp()
tahn = Tahn()
quasi = Quasi()
hyper_params = {
    "activation_functions": [tahn, quasi, quasi],
    "learning_rate": 0.5,
    "weight_mean": 0.0,
    "weight_variance": 1
}

max_epoch = 400
repetitions = 20
success_window = 5
expname = 'arch1 parity7'

inputs, labels = paritaMinus(p)

hidden_sizes = [[p, 5, 3, 1], [p, 5, 15, 1], [p, 10, 15, 1], [p, 15, 10, 1], [p, 15, 15, 1],
                [p, 15, 10, 1], [p, 15, 5, 1], [p, 10, 5, 1], [p, 20, 20, 1], [p, 25, 25, 1],
                [p, 30, 30, 1], [p, 40, 20, 1], [p, 20, 40, 1], [p, 40, 40, 1]]

plot_nets = []
plot_epcs = []

for h in hidden_sizes:
    print("Testing hidden size: {}".format(h))
    hyper_params["layers"] = h
    results = convergence_parity(NASA, hyper_params, repetitions, max_epoch, success_window, inputs, labels, True)
    print(results["epochs"])
    print("Arch1 nets: {}/{} in {} +- {} epochs. Runtime: {:.1f}s \n".format(
        results["nets"],
        repetitions,

        mean(results["epochs"]),
        stdev(results["epochs"]),
        results["time"]
    ))
    plot_nets.append("{} {}\n".format(h, results["nets"]))
    plot_epcs.append(("{} {} {}\n".format(h, mean(results["epochs"]), stdev(results["epochs"]))))

save_results("Arch3", expname, plot_nets, plot_epcs)

# [ 13, 10, 34, 15, 14, 7, 11, 17, 18, 14, 7, 8, 13, 16, 20, 7, 10, 30, 7]
