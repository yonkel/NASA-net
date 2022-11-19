import time
from NASA_one_layer_only import NASA_one_layer
from perceptron import Perceptron
from net_util import Exp, Tahn, SigmoidNp
from generator import paritaMinus, parita
from converg import convergencia
from statistics import mean, stdev
from util import save_results
from util import load_json_parameters

exp = Exp()
tahn = Tahn()
sigmoid = SigmoidNp()

p = 4
expname = 'parity{}_hidden'.format(p)
if p == 2:
    expname = 'xor_hidden'
inputs_minus, labels_minus = paritaMinus(p)
inputs_binary, labels_binary = parita(p)
learning_rate = 0.9
max_epoch = 1000
repetitions = 300
success_window = 10

hidden_size = [9,10,11,12,15,18,21,25]
plot_expnet_nets = []
plot_expnet_epcs = []
plot_mlp_nets = []
plot_mlp_epcs = []

exp_start = time.time()





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

exp_end = time.time()
runtime = exp_end - exp_start
m, s = divmod(runtime, 60)
h, m = divmod(m, 60)
print(s)
print('\nExperiment finished in {:d}:{:02d}:{:02d}'.format(int(h), int(m), round(s)))


save_results("mulnet", expname, plot_expnet_nets, plot_expnet_epcs)
save_results("mlp", expname, plot_mlp_nets, plot_mlp_epcs)



# with open('results/mulnet_{}_nets.txt'.format(expname), 'w') as f:
#     f.write('x y\n')
#     f.writelines(plot_expnet_nets)
# with open('results/mulnet_{}_epcs.txt'.format(expname), 'w') as f:
#     f.write('x y err\n')
#     f.writelines(plot_expnet_epcs)
# with open('results/mlp_{}_nets.txt'.format(expname), 'w') as f:
#     f.write('x y\n')
#     f.writelines(plot_mlp_nets)
# with open('results/mlp_{}_epcs.txt'.format(expname), 'w') as f:
#     f.write('x y err\n')
#     f.writelines(plot_mlp_epcs)
