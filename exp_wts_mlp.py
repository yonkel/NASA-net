import time
from expnet_numpy import ExpNet
from perceptron_numpy import Perceptron
from net_util import Exp, Tahn, SigmoidNp
from generator import paritaMinus, parita
from converg import convergencia
from statistics import mean, stdev
from util import save_results, log_time

p = 2
hidden_size = 2
max_epoch = 1000
repetitions = 100
sigmoid = SigmoidNp()
learning_rate = 0.9
expname = 'parity{}_hidden'.format(p)
if p == 2:
    expname = 'xor_hidden'
inputs_minus, labels_minus = paritaMinus(p)
inputs_binary, labels_binary = parita(p)
success_window = 10
plot_mlp_nets = []
plot_mlp_epcs = []
exp_start = time.time()
for h in hidden_size:
    print("Testing hidden size: {}".format(h))
    architecture = [p, h, 1]
    results_mlp = convergencia(architecture, Perceptron, [sigmoid, sigmoid], learning_rate, max_epoch, repetitions,
                               success_window, inputs_binary, labels_binary, False)
    print("MLP nets: {}/{} in {} +- {} epochs. Runtime: {:.1f}s".format(
        results_mlp["nets"],
        repetitions,
        mean(results_mlp["epochs"]),
        stdev(results_mlp["epochs"]),
        results_mlp["time"]
    ))
    plot_mlp_nets.append("{} {}\n".format(h, results_mlp["nets"]))
    plot_mlp_epcs.append(("{} {} {}\n".format(h, mean(results_mlp["epochs"]), stdev(results_mlp["epochs"]))))
log_time(exp_start)
save_results("mlp", expname, plot_mlp_nets, plot_mlp_epcs)