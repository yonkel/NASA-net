import time
from NASA_one_layer_only import NASA_one_layer
from net_util import Exp, Tahn
from generator import paritaMinus
from converg import convergencia
from statistics import mean, stdev
from util import save_results
p = 2
hidden_size = [2]
max_epoch = 1000
repetitions = 10

exp = Exp()
tahn = Tahn()
expname = 'parity{}_hidden'.format(p)
if p == 2:
    expname = 'xor_hidden'
inputs_minus, labels_minus = paritaMinus(p)
learning_rate = 0.9
success_window = 10
plot_expnet_nets = []
plot_expnet_epcs = []

for h in hidden_size:
    print("Testing hidden size: {}".format(h))
    architecture = [p, h, 1]
    results_expnet = convergencia(architecture, NASA_one_layer, [tahn, exp], learning_rate, max_epoch, repetitions,
                                  success_window, inputs_minus, labels_minus, False)
    print("EXP nets: {}/{} in {} +- {} epochs. Runtime: {:.1f}s".format(
        results_expnet["nets"],
        repetitions,
        mean(results_expnet["epochs"]),
        stdev(results_expnet["epochs"]),
        results_expnet["time"]
    ))
    plot_expnet_nets.append("{} {}\n".format(h, results_expnet["nets"]))
    plot_expnet_epcs.append("{} {} {}\n".format(h, mean(results_expnet["epochs"]), stdev(results_expnet["epochs"])))

# save_results("mulnet", expname, plot_expnet_nets, plot_expnet_epcs)