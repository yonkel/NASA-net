from decimal import Decimal

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold


def parita(n):
    inputs, labels = [], []

    inputs = [list(bin(x)[2:].rjust(n, '0')) for x in range(2 ** n)]
    for i in range(len(inputs)):
        inputs[i] = list(map(int, inputs[i]))
        if inputs[i].count(1) % 2 == 0:
            labels.append([0])
        else:
            labels.append([1])

    return inputs, labels


def paritaMinus(n):
    inputs, labels = [], []

    inputs = [list(bin(x)[2:].rjust(n, '0')) for x in range(2 ** n)]
    for i in range(len(inputs)):
        inputs[i] = list(map(int, inputs[i]))
        if inputs[i].count(1) % 2 == 0:
            labels.append([-1])
        else:
            labels.append([1])

        for j in range(len(inputs[i])):
            if inputs[i][j] == 0:
                inputs[i][j] = -1

    return inputs, labels


def twospirals_raw(n_points, noise=0.5, step_test=5):
    """
     Returns the two spirals dataset.
    """
    n_train = np.sqrt(np.random.rand(n_points, 1)) * 780 * (2 * np.pi) / 360
    n_test = n_train[::step_test]

    n_train = np.delete(n_train, np.arange(0, n_train.size, step_test))
    n_train = n_train.reshape((len(n_train), 1))

    x_train = -np.cos(n_train) * n_train + np.random.rand(n_train.size, 1) * noise
    y_train = np.sin(n_train) * n_train + np.random.rand(n_train.size, 1) * noise

    x_test = -np.cos(n_test) * n_test + np.random.rand(n_test.size, 1) * noise
    y_test = np.sin(n_test) * n_test + np.random.rand(n_test.size, 1) * noise

    train_data = (np.vstack((np.hstack((x_train, y_train)), np.hstack((-x_train, -y_train)))),
                  np.hstack((np.zeros(n_train.size), np.ones(n_train.size))))

    test_data = (np.vstack((np.hstack((x_test, y_test)), np.hstack((-x_test, -y_test)))),
                 np.hstack((np.zeros(n_test.size), np.ones(n_test.size))))

    return train_data, test_data


def twospirals(n_points, step_test = 5):
    n_train = list(range(n_points))
    n_test = n_train[::step_test]
    del n_train[::step_test]

    x_train = []
    for n in n_train:
        r =  0.4 * (105-n) / 104
        a = np.pi * (n-1) / 16
        x1 = r * np.sin(a) + 0.5
        x2 = r * np.cos(a) + 0.5

        x_train.append(np.array([x1, x2]))
        x_train.append(np.array([1-x1, 1-x2]))
    y_train = [i % 2 for i in range(len(x_train))]

    x_test = []
    for n in n_test:
        r = 0.4 * (105 - n) / 104
        a = np.pi * (n - 1) / 16
        x1 = r * np.sin(a) + 0.5
        x2 = r * np.cos(a) + 0.5

        x_test.append(np.array([x1, x2]))
        x_test.append(np.array([1-x1, 1-x2]))
    y_test = [i % 2 for i in range(len(x_test))]

    train = [x_train, y_train]
    test = [x_test, y_test]

    return train, test

def spirals(points, test_batch_size=0.2):
    x, y = twospirals_raw(points)
    y = np.reshape(y, (len(y), 1))

    return x, y


def spiralsMinus(points, test_batch_size=0.2):
    x, y = twospirals_raw(points)

    y = np.where(y == 0, -1, y)
    y = np.reshape(y, (len(y), 1))

    return x, y


def spiralsMinusTransformed(points):
    x, y = twospirals_raw(points)

    m = np.max(x)
    x = x / m

    y = np.where(y == 0, -1, y)
    y = np.reshape(y, (len(y), 1))

    return x, y


def banana():
    inputs, labels = [], []
    with open("banana_dataset.arff") as file:
        for riadok in file:
            riadok = riadok.split(",")
            inputs.append([float(item) for item in riadok[:2]])
            if int(riadok[2]) == 2:
                labels.append([-1])
            else:
                labels.append([1])

    return inputs, labels


def banana_Transformed():
    inputs, labels = [], []
    with open("banana_dataset.arff") as file:
        for riadok in file:
            riadok = riadok.split(",")
            inputs.append([float(item) for item in riadok[:2]])
            if int(riadok[2]) == 2:
                labels.append([-1])
            else:
                labels.append([1])

    m = np.max(inputs)
    inputs = inputs / m

    return inputs, labels


if __name__ == "__main__":
    pass
    train, test = twospirals(100)

    # print("?")
    #

    for data in train, test:
        x, y = data

        for i in range(len(x)):
            if y[i] == 1:
                plt.scatter(x[i][0], x[i][1], color="red")
            else:
                plt.scatter(x[i][0], x[i][1], color="blue")

        plt.show()
