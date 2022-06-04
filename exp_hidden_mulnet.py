import time
from expnet_numpy import ExpNet
from net_util import Exp, Tahn
from generator import paritaMinus
from converg import convergencia
from statistics import mean, stdev
from util import save_results, log_time

p = 2
hidden_size = [2,4]
max_epoch = 1000
repetitions = 100

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
exp_start = time.time()
for h in hidden_size:
    print("Testing hidden size: {}".format(h))
    architecture = [p, h, 1]
    results_expnet = convergencia(architecture, ExpNet, [tahn, exp], learning_rate, max_epoch, repetitions,
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
log_time(exp_start)
save_results("mulnet", expname, plot_expnet_nets, plot_expnet_epcs)