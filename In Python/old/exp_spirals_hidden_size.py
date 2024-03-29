import time
from NASA_one_layer_only import NASA_one_layer
from perceptron import Perceptron
from net_util import Exp, Tahn, SigmoidNp
from convergence_general import convergence_general
from statistics import mean, stdev
from util import save_results
from generator import spirals, spiralsMinus, spiralsMinusTransformed


exp = Exp()
tahn = Tahn()
sigmoid = SigmoidNp()


expname = 'spirals_hidden'
learning_rate = 0.2
max_epoch = 100
repetitions = 2

hidden_size = [10]
plot_expnet_nets = []
plot_expnet_epcs = []

exp_start = time.time()
wanted_MSE = 0.1
spirals_data = spiralsMinusTransformed(500)

save_params = {
    "exp_name" : "spirals_hidden",
    "net_name" : "mulnet",
}



for h in hidden_size:
    print("Testing hidden size: {}".format(h))
    architecture = [2, h, 1]
    save_params["value"] = h
    results_expnet = convergence_general(architecture, NASA_one_layer, [tahn, exp], learning_rate, max_epoch, repetitions,
                                         wanted_MSE, spirals_data, False, save_params)
    print("EXP nets: {}/{} in {} +- {} epochs. Runtime: {:.1f}s".format(
        results_expnet["nets"],
        repetitions,
        mean(results_expnet["epochs"]),
        stdev(results_expnet["epochs"]),
        results_expnet["time"]
    ))
    plot_expnet_nets.append("{} {}\n".format(h, results_expnet["nets"]))
    plot_expnet_epcs.append("{} {} {}\n".format(h, mean(results_expnet["epochs"]), stdev(results_expnet["epochs"])))


exp_end = time.time()
runtime = exp_end - exp_start
m, s = divmod(runtime, 60)
h, m = divmod(m, 60)
print(s)
print('\nExperiment finished in {:d}:{:02d}:{:02d}'.format(int(h), int(m), round(s)))


save_results("mulnet", expname, plot_expnet_nets, plot_expnet_epcs)

