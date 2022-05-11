from converg import convergencia
from expnet_numpy import ExpNet
from net_util import Exp, Tahn
exp = Exp()
tahn = Tahn()
from generator import paritaMinus
import matplotlib.pyplot as plt

files = "w" # "a"  --- w for overwriting files, a for appending

p = 2
inputs, labels = paritaMinus(p)

architecture = [p,2,1]

learning_rate = 0.5
max_epoch = 1000
repetitions = 100
success_window = 10
net_type = ExpNet

show = False
# True it you want "vypis do konzoly" and graphs for every convergence run



def generate_for_h( indexes, filename=None ):
    results = []
    for h in indexes:
            architecture[1] = h

            results.append([h, *convergencia( architecture, net_type, learning_rate, max_epoch, repetitions, success_window, inputs, labels, show)])

    # plt.plot(list(indexes), [item[1] for item in results])
    # plt.show()

    if filename:
        with open(f"{filename}_success_rate.txt", files) as file:
            file.write(f"hidden_size success_rate\n")
            for item in results:
                file.write(f"{item[0]} {item[1]}\n")

        with open(f"{filename}_training_epochs.txt", files) as file:
            file.write(f"hidden_size success_rate\n")
            for item in results:
                file.write(f"{item[0]} {item[2]}\n")

    return results



def generate_for_p( indexes, filename=None ):
    results = []
    for P in indexes:
        _inputs, _labels = paritaMinus(P)
        architecture[0] = P
        results.append([P, *convergencia( architecture, net_type, learning_rate, max_epoch, repetitions, success_window, _inputs, _labels, show)])

    # plt.plot(list(indexes), [item[2] for item in results])
    # plt.show()

    if filename:
        with open(f"{filename}_success_rate.txt", files) as file:
            file.write(f"parity success_rate\n")
            for item in results:
                file.write(f"{item[0]} {item[1]}\n")

        with open(f"{filename}_training_epochs.txt", files) as file:
            file.write(f"parity success_rate\n")
            for item in results:
                file.write(f"{item[0]} {item[2]}\n")

    return results





if __name__ == "__main__":
    pass
    # convergencia( architecture, net_type, learning_rate, max_epoch, repetitions, success_window, inputs, labels, show )

    res = generate_for_h(range(2, 4), "hidden")
    print(res)

    res_p = generate_for_p(range(2,3), "parity" )
    print(res_p)
