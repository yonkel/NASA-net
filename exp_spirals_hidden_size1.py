import time
from expnet_numpy import ExpNet
from perceptron_numpy import Perceptron
from net_util import Exp, Tahn, SigmoidNp
from convergence_general import convergence_general
from statistics import mean, stdev
from util import save_results
from generator import spirals, spiralsMinus


exp = Exp()
tahn = Tahn()
sigmoid = SigmoidNp()


expname = 'spirals_hidden'
learning_rate = 0.02
max_epoch = 10000
repetitions = 5

hidden_size = [20]
plot_expnet_nets = []
plot_expnet_epcs = []
plot_expnet_mses = []

exp_start = time.time()
wanted_MSE = 0.1
spirals_data = spirals(500)




for h in hidden_size:
    print("Testing hidden size: {}".format(h))
    architecture = [2, h, 1]
    results_expnet = convergence_general( architecture, ExpNet, [tahn, exp], learning_rate, max_epoch, repetitions,
                                           wanted_MSE , spirals_data , False )
    print("EXP nets: {}/{} in {} +- {} epochs. Runtime: {:.1f}s".format(
        results_expnet["nets"],
        repetitions,
        mean(results_expnet["epochs"]),
        stdev(results_expnet["epochs"]),
        results_expnet["time"]
    ))
    plot_expnet_nets.append("{} {}\n".format(h, results_expnet["nets"]))
    plot_expnet_epcs.append("{} {} {}\n".format(h, mean(results_expnet["epochs"]), stdev(results_expnet["epochs"])))
    plot_expnet_mses.append("{} {} \n".format(h, results_expnet["mse"]))

exp_end = time.time()
runtime = exp_end - exp_start
m, s = divmod(runtime, 60)
h, m = divmod(m, 60)
print(s)
print('\nExperiment finished in {:d}:{:02d}:{:02d}'.format(int(h), int(m), round(s)))


save_results("mulnet", expname, plot_expnet_nets, plot_expnet_epcs, plot_expnet_mses)

