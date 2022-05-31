from decimal import Decimal

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

def parita(n):
    inputs, labels = [],[]

    inputs = [list(bin(x)[2:].rjust(n, '0')) for x in range(2 ** n)]
    for i in range(len(inputs)):
        inputs[i] = list(map(int, inputs[i]))
        if inputs[i].count(1) % 2 == 0 :
            labels.append([0])
        else:
            labels.append([1])

    return inputs, labels


def paritaMinus(n):
    inputs, labels = [],[]

    inputs = [list(bin(x)[2:].rjust(n, '0')) for x in range(2 ** n)]
    for i in range(len(inputs)):
        inputs[i] = list(map(int, inputs[i]))
        if inputs[i].count(1) % 2 == 0 :
            labels.append([-1])
        else:
            labels.append([1])

        for j in range(len(inputs[i])):
            if inputs[i][j] == 0:
                inputs[i][j] = -1

    return inputs, labels



def twospirals_raw(n_points, noise=.5):
    """
     Returns the two spirals dataset.
    """
    n = np.sqrt(np.random.rand(n_points,1)) * 780 * (2*np.pi)/360
    d1x = -np.cos(n)*n + np.random.rand(n_points,1) * noise
    d1y = np.sin(n)*n + np.random.rand(n_points,1) * noise
    return (np.vstack((np.hstack((d1x,d1y)),np.hstack((-d1x,-d1y)))),
            np.hstack((np.zeros(n_points),np.ones(n_points))))


def spirals(points, test_batch_size=0.2):
    x, y = twospirals_raw(points)
    data_train, data_test, labels_train, labels_test =  train_test_split(x, y, test_size=test_batch_size)

    labels_train = np.reshape(labels_train, (len(labels_train),1))
    labels_test = np.reshape(labels_test, (len(labels_test),1))

    return data_train, data_test, labels_train, labels_test

def spiralsMinus(points, test_batch_size=0.2):
    x, y = twospirals_raw(points)

    data_train, data_test, labels_train, labels_test = train_test_split(x, y, test_size=test_batch_size)

    labels_train = np.where(labels_train == 0, -1, labels_train)
    labels_train = np.reshape(labels_train, (len(labels_train), 1))

    labels_test = np.where(labels_test == 0, -1, labels_test)
    labels_test = np.reshape(labels_test, (len(labels_test), 1))


    return data_train, data_test, labels_train, labels_test


def banana():
    inputs, labels = [], []
    with open("banana_dataset.arff") as file:
        for riadok in file:
            riadok = riadok.split(",")
            inputs.append( [ float(item) for item in riadok[:2]]  )
            if int(riadok[2]) == 2:
                labels.append([-1])
            else:
                labels.append([1])

    return  train_test_split(inputs, labels, test_size=0.2)
    # data_train, data_test, labels_train, labels_test =
    # return data_train + data_test , data_test, labels_train + labels_test , labels_test

if __name__ == "__main__":
    pass
    # x, y = twospirals(500)
    # plt.title('training set')
    # plt.plot(x[y == 0, 0], x[y == 0, 1], '.', label='class 1')
    # plt.plot(x[y == 1, 0], x[y == 1, 1], '.', label='class 2')
    # plt.legend()
    # plt.show()

    # x, x_t, y, y_t = banana()
    # for i in range(len(x)):
    #     print(x[i], y[i])
    # print(len(x))
    # print(len(x_t))
    # print(len(x_t)/len(x))
    # 5300
    # print(inp)
    # print(lab)

