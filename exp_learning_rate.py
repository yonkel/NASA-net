import time
from expnet_numpy import ExpNet
from perceptron_numpy import Perceptron
from net_util import Exp, Tahn, SigmoidNp
from generator import paritaMinus, parita
from converg import convergencia
from statistics import mean, stdev
from util import save_results
from util import load_json_parameters


exp = Exp()
tahn = Tahn()
sigmoid = SigmoidNp()

p = 2
expname = 'parity{}_lr'.format(p)
if p == 2:
    expname = 'xor_lr'
inputs_minus, labels_minus = paritaMinus(p)
inputs_binary, labels_binary = parita(p)
hidden_size = 2
max_epoch = 600
repetitions = 5
success_window = 10

learning_rate = [0.1,0.3,0.5,0.7,0.9,1.0,1.2]
plot_expnet_nets = []
plot_expnet_epcs = []
plot_mlp_nets = []
plot_mlp_epcs = []

# parameters = load_json_parameters()
# p = parameters["p"]
# hidden_size = parameters["hidden_size"]
# max_epoch = parameters["max_epoch"]
# repetitions = parameters["repetitions"]
# success_window = parameters["success_window"]
# learning_rate = parameters["learning_rate"]
#
# print( p  )
# print(  hidden_size )
# print(  max_epoch )
# print( repetitions  )
# print(  success_window )
# print(  learning_rate )


exp_start = time.time()

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
    plot_expnet_epcs.append("{} {} {}\n".format(lr, mean(results_expnet["epochs"]), stdev(results_expnet["epochs"])))
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
    plot_mlp_epcs.append(("{} {} {}\n".format(lr, mean(results_mlp["epochs"]), stdev(results_mlp["epochs"]))))

exp_end = time.time()
runtime = exp_end - exp_start
m, s = divmod(runtime, 60)
h, m = divmod(m, 60)
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


