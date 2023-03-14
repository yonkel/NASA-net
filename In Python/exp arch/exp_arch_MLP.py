import time
from NASA import NASA
from net_util import Quasi, Tahn, SigmoidNp
from generator import parita
from Convergence_new import convergence
from statistics import mean, stdev
from util import save_results
from sklearn.model_selection import KFold




# set network PARAMS
sigm = SigmoidNp()
tahn = Tahn()
quasi = Quasi()
hyper_params = {
    "activation_functions": [sigm, sigm, sigm],
    "learning_rate": 0.5,
    "weight_mean": 0.0,
    "weight_variance": 1
}



max_epoch = 1000
repetitions = 20
success_window = 5
expname = 'arch1 parity7'

p = 7
inputs, labels = parita(p)

hidden_sizes = [[p, 50, 50, 1], [p, 65, 65, 1], [p, 85, 85, 1], [p, 100, 100, 1], [p, 120, 120, 1]]

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

save_results("MLP", expname, plot_nets, plot_epcs)


