import json
import sys
import time


def log_time(exp_start):
    now = time.time()
    runtime = now - exp_start
    m, s = divmod(runtime, 60)
    h, m = divmod(m, 60)
    print('\nExperiment finished in {:d}:{:02d}:{:02d}'.format(int(h), int(m), round(s)))


def save_results( net_name, exp_name, nets, epc):
    with open(f'results/{net_name}_{exp_name}_nets.txt', 'w') as f:
        f.write('x y\n')
        f.writelines(nets)
    with open(f'results/{net_name}_{exp_name}_epcs.txt', 'w') as f:
        f.write('x y err\n')
        f.writelines(epc)


def generate_parameters_hidden():
    parameters = {
        'parameters_mlp': {
            'p': 2,
            'learning_rate': 0.9,
            'max_epoch': 1000,
            'repetitions': 100,
            'success_window': 10,
            'hidden_size': [2, 4]
        },
        'parameters_mulnet': {
            'p': 2,
            'learning_rate': 0.9,
            'max_epoch': 1000,
            'repetitions': 100,
            'success_window': 10,
            'hidden_size': [2, 4]
        }
    }

    with open('exp_hidden_parameters.json', 'w') as outfile:
        json_string = json.dumps(parameters, indent=4)
        outfile.write(json_string)


def generate_parameters_lr():
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

    with open('exp_lr_parameters.json', 'w') as outfile:
        json_string = json.dumps(parameters, indent=4)
        outfile.write(json_string)


def load_json_parameters():
    pass

#TODO 1. load funkcia pre Json, implementacia citania Jsonu
#TODO 2. zovseobecnit generujucu funkciu, vytvorit parametre pre kazdu premm
#TODO 3. napisat funkciu ktora bude podobna tymto ( prehladne sa menia parametre ), ktora na zaver zavola vseobecnu


# co treba : treba pripravit funkciu na citanie parametrov z jsonu -  nazov json suboru nacita z konzoly
#            zmenit experimenty?, urobit 2 ?, zajtra zavolat kike...



if __name__ == '__main__':
    pass

    generate_parameters_lr()

    with open('exp_lr_parameters.json') as json_file:
        data = json.load(json_file)

        print(data['parameters_mlp']['hidden_size'])

