import random
import numpy as np
import matplotlib.pyplot as plt
from expnet_numpy import ExpNet
from perceptron_numpy import Perceptron
from net_util import Exp, Tahn, SigmoidNp
from generator import paritaMinus
from converg import convergencia
from statistics import mean, stdev

exp = Exp()
tahn = Tahn()
sigmoid = SigmoidNp()

p = 2
inputs, labels = paritaMinus(p)
learning_rate = 0.5
max_epoch = 1000
repetitions = 10
success_window = 10

hidden_size = [2,3,4,5,6,7,8]
results = {}
for h in hidden_size:
    print("Testing hidden size: {}".format(h))
    architecture = [p, h, 1]
    results_expnet = convergencia(architecture, ExpNet, [tahn, exp], learning_rate, max_epoch, repetitions, success_window, inputs, labels, False)
    print("EXP nets: {}/{} in {} +- {} epochs. Runtime: {:.1f}s".format(
        results_expnet["nets"],
        repetitions,
        mean(results_expnet["epochs"]),
        stdev(results_expnet["epochs"]),
        results_expnet["time"]
    ))
    results_mlp = convergencia(architecture, Perceptron, [sigmoid, sigmoid], learning_rate, max_epoch, repetitions, success_window, inputs, labels, False)
    print("MLP nets: {}/{} in {} +- {} epochs. Runtime: {:.1f}s".format(
        results_mlp["nets"],
        repetitions,
        mean(results_mlp["epochs"]),
        stdev(results_mlp["epochs"]),
        results_expnet["time"]
    ))

