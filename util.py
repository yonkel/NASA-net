import json
import sys
from decimal import *
import numpy as np
import scipy.special as sc
from net_util import Tahn


def save_results( net_name, exp_name, nets, epc, mse=None):
    with open(f'results/{net_name}_{exp_name}_nets.txt', 'a') as f:
        f.write('x y\n')
        f.writelines(nets)
    with open(f'results/{net_name}_{exp_name}_epcs.txt', 'a') as f:
        f.write('x y err\n')
        f.writelines(epc)
    if mse != None:
        with open(f'results/{net_name}_{exp_name}_mses.txt', 'a') as f:
            f.write('x y\n')
            f.writelines(epc)


def generate_parameters( experiment_name ):
    parameters = {
        'parameters': {
            'p': 2,
            'hidden_size': 2,
            'max_epoch': 1000,
            'repetitions': 100,
            'success_window': 10,
            'learning_rate': [0.3, 0.5, 0.7, 0.9, 1.0, 1.2]
        }
    }

    with open(f'exp_{experiment_name}_parameters.json', 'w') as outfile:
        json_string = json.dumps(parameters, indent=4)
        outfile.write(json_string)



def generate_different_parameters( experiment_name ):
    parameters = {
        'parameters_mlp': {
            'p': 2,
            'hidden_size': 2,
            'max_epoch': 1000,
            'repetitions': 100,
            'success_window': 10,
            'learning_rate': [0.1, 0.3, 0.5, 0.7, 0.9, 1.0, 1.2]
        },
        'parameters_mulnet': {
            'p': 2,
            'hidden_size': 2,
            'max_epoch': 1000,
            'repetitions': 100,
            'success_window': 10,
            'learning_rate': [0.1, 0.3, 0.5, 0.7, 0.9, 1.0, 1.2]
        }
    }

    with open(f'exp_{experiment_name}_parameters.json', 'w') as outfile:
        json_string = json.dumps(parameters, indent=4)
        outfile.write(json_string)


def load_json_parameters():
    file = str(sys.argv[1])
    with open(file) as json_file:
        parameters = json.load(json_file)

        if len(parameters.keys() )> 1:
            par_mlp = parameters['parameters_mlp']
            par_mulnnet = parameters['parameters_mulnet']
            return par_mlp, par_mulnnet
        else:
            parameters = parameters['parameters']
            return parameters

#TODO co keby sa parametre pre experimenty posuvali v slovniku



if __name__ == '__main__':
    pass

    tahn = Tahn()

    x = np.array([
                 [  72.64023382],
                 [-658.37294647],
                 [  65.90698254],
                 [ -58.11132943],
                 [   3.29965272],
                 [-713.01799043],
                 [ -79.22565105],
                 [ -37.80401742],
                 ])

    # y = []
    # for item in x:
    #     y.append(np.asarray(item, dtype = np.longdouble))
    #
    # y = np.asarray(y, dtype = np.longdouble )
    #
    # print(type(y[0][0]))
    #
    # np.exp(-y)



    # x = np.asarray([[Decimal(el[0]), Decimal(el[1])] for el in x], dtype=object)

    print(tahn.apply_func(x))
    print()

    for i in range(len(x)):
        if x[i][0] > 709 or x[i][0] < -709:
            x = np.asarray([[Decimal(el[0])] for el in x], dtype=object)
            break
    nieco = ( np.exp(x) - np.exp(-x) ) / ( np.exp(x) + np.exp(-x) )
    pole = np.asarray(nieco, dtype=float)

    print(pole)





