import time
from expnet_numpy import ExpNet
from perceptron_numpy import Perceptron
from net_util import Exp, Tahn, SigmoidNp
from convergence_twospirals import convergencia_spirals
from statistics import mean, stdev
from util import save_results
from util import load_json_parameters


exp = Exp()
tahn = Tahn()
sigmoid = SigmoidNp()


expname = 'spirals_hidden'
learning_rate = 0.9
max_epoch = 5000
repetitions = 50

hidden_size = [6, 9, 15, 20, 25, 30]
plot_expnet_nets = []
plot_expnet_epcs = []

exp_start = time.time()
wanted_MSE = 0.5
spiral_nodes = 500




for h in hidden_size:
    print("Testing hidden size: {}".format(h))
    architecture = [2, h, 1]
    results_expnet = convergencia_spirals( architecture, ExpNet, [tahn, exp], learning_rate, max_epoch, repetitions,
                                           wanted_MSE , spiral_nodes , False )
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

