import time
from NASA import ExpNet
from perceptron import Perceptron
from net_util import Exp, Tahn, SigmoidNp
from generator import paritaMinus, parita
from converg import convergencia
from statistics import mean, stdev
from util import save_results,log_time

exp = Exp()
tahn = Tahn()
sigmoid = SigmoidNp()

p = 3
expname = 'parity{}_wts_variance'.format(p)
if p == 2:
    expname = 'xor_wts_variance'
inputs_minus, labels_minus = paritaMinus(p)
inputs_binary, labels_binary = parita(p)
learning_rate = 0.9
max_epoch = 3000
repetitions = 100
success_window = 10
hidden_size = 12

# wts_variance = [0.2,0.5,0.7,0.9,1.0,1.2,1.5,2.0,2.5]
# wts_variance = [0.1,0.2,0.3,0.4,0.5,0.6,0.7]
wts_variance = [0.1,0.3]

plot_expnet_nets = []
plot_expnet_epcs = []
plot_mlp_nets = []
plot_mlp_epcs = []

exp_start = time.time()

for var in wts_variance:
    print("Testing variance: {}".format(var))
    architecture = [p, hidden_size, 1]
    results_expnet = convergencia(architecture, ExpNet, [tahn, exp], learning_rate, max_epoch, repetitions,
                                  success_window, inputs_minus, labels_minus, False, wts_variance=var)
    print("EXP nets: {}/{} in {} +- {} epochs. Runtime: {:.1f}s".format(
        results_expnet["nets"],
        repetitions,
        mean(results_expnet["epochs"]),
        stdev(results_expnet["epochs"]),
        results_expnet["time"]
    ))
    plot_expnet_nets.append("{} {}\n".format(var, results_expnet["nets"]))
    plot_expnet_epcs.append("{} {} {}\n".format(var, mean(results_expnet["epochs"]), stdev(results_expnet["epochs"])))
    results_mlp = convergencia(architecture, Perceptron, [sigmoid, sigmoid], learning_rate, max_epoch, repetitions,
                               success_window, inputs_binary, labels_binary, False, wts_variance=var)
    print("MLP nets: {}/{} in {} +- {} epochs. Runtime: {:.1f}s".format(
        results_mlp["nets"],
        repetitions,
        mean(results_mlp["epochs"]),
        stdev(results_mlp["epochs"]),
        results_mlp["time"]
    ))
    plot_mlp_nets.append("{} {}\n".format(var, results_mlp["nets"]))
    plot_mlp_epcs.append(("{} {} {}\n".format(var, mean(results_mlp["epochs"]), stdev(results_mlp["epochs"]))))
log_time(exp_start)

save_results("mulnet", expname, plot_expnet_nets, plot_expnet_epcs)
save_results("mlp", expname, plot_mlp_nets, plot_mlp_epcs)