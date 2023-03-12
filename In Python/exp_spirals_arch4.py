import time
from NASA import NASA
from net_util import Quasi, Tahn, SigmoidNp
from generator import paritaMinus
from Convergence_new import convergence
from statistics import mean, stdev
from util import save_results

inputs, labels, z, zz = spiralsMinusTransformed(200)

# set network PARAMS
tahn = Tahn()
quasi = Quasi()

hyper_params = {
    "activation_functions": [quasi, quasi, tahn],
    "learning_rate": 0.005,
    "weight_mean": 0.0,
    "weight_variance": 1
}

max_epoch = 400
repetitions = 20
success_window = 5
expname = 'spirals'

x_dim = 2

hidden_sizes = [[x_dim, 5, 3, 1], [x_dim, 5, 10, 1], [x_dim, 5, 15, 1], [x_dim, 10, 15, 1], [x_dim, 15, 10, 1], [x_dim, 15, 15, 1],
                [x_dim, 15, 10, 1], [x_dim, 15, 5, 1], [x_dim, 10, 5, 1]]

plot_nets = []
plot_epcs = []

for h in hidden_sizes:
    print("Testing hidden size: {}".format(h))
    hyper_params["layers"] = h
    results = convergence(NASA, hyper_params, repetitions, max_epoch, success_window, inputs, labels, True)
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

save_results("Arch4", expname, plot_nets, plot_epcs)