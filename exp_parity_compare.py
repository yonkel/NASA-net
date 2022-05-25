import time
from expnet_numpy import ExpNet
from perceptron_numpy import Perceptron
from net_util import Exp, Tahn, SigmoidNp
from generator import paritaMinus, parita
from converg import convergencia
from statistics import mean, stdev
from util import save_results, log_time

expname = 'parity_compare'
max_parity = 5
hidden_size_expnet = {3:4, 4:6, 5:12, 6:12, 7:15}
hidden_size_mlp = {3:9, 4:12, 5:50}
learning_rate = 0.9
max_epoch = 1000
repetitions = 100
success_window = 10
exp = Exp()
tahn = Tahn()
sigmoid = SigmoidNp()
plot_expnet_nets = []
plot_expnet_epcs = []
plot_mlp_nets = []
plot_mlp_epcs = []
exp_start = time.time()
for p in range(2,max_parity+1):
    inputs_minus, labels_minus = paritaMinus(p)
    inputs_binary, labels_binary = parita(p)
    print("Testing parity: {}".format(p))
    results_expnet = convergencia([p, hidden_size_expnet[p], 1], ExpNet, [tahn, exp], learning_rate, max_epoch, repetitions,
                                  success_window, inputs_minus, labels_minus, False)
    print("EXP nets: {}/{} in {} +- {} epochs. Runtime: {:.1f}s".format(
        results_expnet["nets"],
        repetitions,
        mean(results_expnet["epochs"]),
        stdev(results_expnet["epochs"]),
        results_expnet["time"]
    ))
    plot_expnet_nets.append("{} {}\n".format(p, results_expnet["nets"]))
    plot_expnet_epcs.append("{} {} {}\n".format(p, mean(results_expnet["epochs"]), stdev(results_expnet["epochs"])))
    results_mlp = convergencia([p, hidden_size_mlp[p], 1], Perceptron, [sigmoid, sigmoid], learning_rate, max_epoch, repetitions,
                               success_window, inputs_binary, labels_binary, False)
    print("MLP nets: {}/{} in {} +- {} epochs. Runtime: {:.1f}s".format(
        results_mlp["nets"],
        repetitions,
        mean(results_mlp["epochs"]),
        stdev(results_mlp["epochs"]),
        results_mlp["time"]
    ))
    plot_mlp_nets.append("{} {}\n".format(p, results_mlp["nets"]))
    plot_mlp_epcs.append(("{} {} {}\n".format(p, mean(results_mlp["epochs"]), stdev(results_mlp["epochs"]))))

log_time(exp_start)
save_results("mulnet", expname, plot_expnet_nets, plot_expnet_epcs)
save_results("mlp", expname, plot_mlp_nets, plot_mlp_epcs)