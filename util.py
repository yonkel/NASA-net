import json
import sys, os
from decimal import *
import numpy as np
import scipy.special as sc
from net_util import Tahn
import os



def save_results( net_name, exp_name, nets, epc, mse=None, dets = None):
    if not os.path.exists("results"):
        os.makedirs("results")
    with open(f'results/{net_name}_{exp_name}_nets.txt', 'a') as f:
        f.write('x y\n')
        f.writelines(nets)
    with open(f'results/{net_name}_{exp_name}_epcs.txt', 'a') as f:
        f.write('x y err\n')
        f.writelines(epc)
    if mse != None:
        with open(f'results/{net_name}_{exp_name}_mses.txt', 'a') as f:
            f.write('x y\n')
            f.writelines(mse)
    if dets != None:
        with open(f'results/{net_name}_{exp_name}_dets.txt', 'a') as f:
            f.write('x y\n')
            f.writelines(dets)



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





