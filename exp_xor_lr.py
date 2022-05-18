import random
import numpy as np
import matplotlib.pyplot as plt
from expnet_numpy import ExpNet
from perceptron_numpy import Perceptron
from net_util import Exp, Tahn, SigmoidNp
from generator import paritaMinus, parita
from converg import convergencia
from statistics import mean, stdev

exp = Exp()
tahn = Tahn()
sigmoid = SigmoidNp()

p = 2
inputs_minus, labels_minus = paritaMinus(p)
inputs_binary, labels_binary = parita(p)
hidden_size = 2
max_epoch = 500
repetitions = 10
success_window = 10

learning_rate = [0.1,0.3,0.5,0.7,0.9,1.0,1.2]
plot_expnet_nets = []
plot_expnet_epc = []
plot_mlp_nets = []
plot_mlp_epc = []
for lr in learning_rate:
    print("Testing learning rate: {}".format(lr))
    architecture = [p, hidden_size, 1]
    results_expnet = convergencia(architecture, ExpNet, [tahn, exp], lr, max_epoch, repetitions,
                                  success_window, inputs_minus, labels_minus, False)
    print("EXP nets: {}/{} in {} +- {} epochs. Runtime: {:.1f}s".format(
        results_expnet["nets"],
        repetitions,
        mean(results_expnet["epochs"]),
        stdev(results_expnet["epochs"]),
        results_expnet["time"]
    ))
    plot_expnet_nets.append("{} {}\n".format(lr, results_expnet["nets"]))
    plot_expnet_epc.append("{} {} {}\n".format(lr, mean(results_expnet["epochs"]), stdev(results_expnet["epochs"])))
    results_mlp = convergencia(architecture, Perceptron, [sigmoid, sigmoid], lr, max_epoch, repetitions,
                               success_window, inputs_binary, labels_binary, False)
    print("MLP nets: {}/{} in {} +- {} epochs. Runtime: {:.1f}s".format(
        results_mlp["nets"],
        repetitions,
        mean(results_mlp["epochs"]),
        stdev(results_mlp["epochs"]),
        results_mlp["time"]
    ))
    plot_mlp_nets.append("{} {}\n".format(lr, results_mlp["nets"]))
    plot_mlp_epc.append(("{} {} {}\n".format(lr, mean(results_mlp["epochs"]), stdev(results_mlp["epochs"]))))

with open('results/mulnet_xor_hidden_nets.txt', 'w') as f:
    f.write('x y\n')
    f.writelines(plot_expnet_nets)
with open('results/mulnet_xor_hidden_epcs.txt', 'w') as f:
    f.write('x y err\n')
    f.writelines(plot_expnet_nets)
with open('results/mlp_xor_hidden_nets.txt', 'w') as f:
    f.write('x y\n')
    f.writelines(plot_mlp_nets)
with open('results/mlp_xor_hidden_epcs.txt', 'w') as f:
    f.write('x y err\n')
    f.writelines(plot_mlp_epc)


