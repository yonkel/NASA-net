import time
from NASA import NASA
from net_util import Quasi, Tahn, SigmoidNp
from generator import paritaMinus
from Convergence_spirals import k_fold_spirals
from statistics import mean, stdev
from util import save_results


# set network PARAMS
tahn = Tahn()
quasi = Quasi()

hyper_params = {
    "activation_functions": [quasi, quasi, tahn],
    "learning_rate": 0.005,
    "weight_mean": 0.0,
    "weight_variance": 1
}

max_epoch = 400
repetitions = 20
success_window = 5
expname = 'spirals'
number_of_points = 200 # 1 spiral == 200 points

x_dim = 2

hidden_sizes = [[x_dim, 10, 10, 1]]

plot_nets = []
plot_epcs = []

for h in hidden_sizes:
    print("Testing hidden size: {}".format(h))
    hyper_params["layers"] = h
    results = k_fold_spirals(NASA, hyper_params, repetitions, max_epoch, number_of_points, True)
    print(f"Arch4 {h} : good outputs = {results['good_outputs']} \n of {results['max_good_test']} "
          f"\nMSE: {results['MSE']}".format(
    ))
    save_MSE("Arch4", expname, good_outputs, MSE)

