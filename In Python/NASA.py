import math
import numpy as np
from net_util import SigmoidNp, Tahn, Quasi
from generator import paritaMinus

class NASA:
    def __init__(self, params):

        # read hyperparameters
        self.layers = params["layers"]  # [ input dimension, h_1, ... , h_n, output dimension ]
        self.activation_functions = params["activation_functions"]
        self.learning_rate = params["learning_rate"]

        self.sigmoid = SigmoidNp()

        # initialize Weights
        self.W = []
        for i in range(len(self.layers) - 1):
            if type(self.activation_functions[i]) == Quasi:
                self.W.append(np.random.normal(params["weight_mean"], params["weight_variance"],
                                               (params["layers"][i + 1], params["layers"][i])))
            else:  # add bias if not Quasi
                self.W.append(np.random.normal(params["weight_mean"], params["weight_variance"],
                                               (params["layers"][i + 1], params["layers"][i] + 1)))

        # for i, w in enumerate(self.W):
        #     print(i, w)
        #     print()
    def activation(self, x):
        h = [x]

        for i in range(len(self.activation_functions)):
            if type(self.activation_functions[i]) == Quasi:
                h_i = self.activation_functions[i].apply_func(self.W[i], h[-1])
            else:
                biased_h_i = np.vstack([h[-1], np.ones(len(h[-1][0]))])
                h_i_dot = self.W[i] @ biased_h_i
                h_i = self.activation_functions[i].apply_func(h_i_dot)

            h.append(h_i)

        # h = [x, h1, ..., hn, y]
        # print(h[-2], h[-2].shape)
        return h

    def quasi_learn(self, error, W):
        pass

    def learning(self, h, d):
        y = h[-1]
        error = [(d - y)]

        w_changes = []

        for i in range(len(self.activation_functions) - 1, -1, -1):

            if type(self.activation_functions[i]) == Quasi:
                error_tmp = h[i].copy()
                w_change_tmp = self.W[i].copy()

                delta = error * h[i + 1]

                for j in range(self.layers[i]):
                    tmp = 0
                    for k in range(self.layers[i + 1]):
                        logistic_degree = self.sigmoid.apply_func(self.W[i][k][j])

                        common_term_j_k = self.activation_functions[i].quasiPow(h[i][j][0], logistic_degree)

                        # print("common_term_j_k", common_term_j_k, common_term_j_k.shape)

                        if common_term_j_k < 0.0000000001:
                            # print("som tu pre ", h[0])
                            common_term_j_k = 1.0
                            for j2 in range(self.layers[i]):
                                if j2 != j:
                                    common_term_j_k *= self.activation_functions[i].quasiPow(h[i][j2][0],
                                                                                             self.sigmoid.apply_func(
                                                                                                 self.W[i][k][j2]))
                            try: #NP arrays :))) hehehehe
                                common_term_j_k *= error[k][0]
                            except:
                                common_term_j_k *= error

                        else:
                            common_term_j_k = delta[k] / common_term_j_k

                        if (np.isnan(common_term_j_k).any()):
                            input("NAN ERROR - cakam")


                        try:
                            w_change_tmp[k][j] = common_term_j_k * (h[i][j] - 1) * logistic_degree * (1 - logistic_degree)  # toto je druhy riadok rovnice
                        except:
                            print("--------ERROR---------")
                            print("w_change_tmp")
                            print(w_change_tmp, w_change_tmp.shape)
                            print("common_term_j_k")
                            print(common_term_j_k * (h[i][j] - 1) * logistic_degree * (1 - logistic_degree))

                        # print("w_change_tmp")
                        # print(w_change_tmp, w_change_tmp.shape)
                        # print("common_term_j_k")
                        # print(common_term_j_k, common_term_j_k.shape)
                        # input("good")

                        tmp += common_term_j_k * logistic_degree

                    error_tmp[j] = tmp

                error = error_tmp
                w_changes.insert(0, w_change_tmp)


            else:
                delta = error * self.activation_functions[i].apply_derived(h[i + 1])

                biased_hi = np.vstack([h[i], np.ones(len(h[i][0]))])
                w_change = delta @ biased_hi.T

                if (self.W[i].shape != w_change.shape):
                    # print("reshape", self.W[i].shape, w_change.shape, print(biased_hi.shape))
                    w_change = w_change[0]

                w_changes.insert(0, w_change)
                error = self.W[i][:, :self.layers[i]].T @ delta
                if error.shape != (self.layers[i+1], 1):
                    error = error[0]
        # a @ b.T == (b @ a.T).T

        # print("weight_change_output", w_changes[1])
        # print()
        # print("weight_change_hidden", w_changes[0])
        # print("\nnew")

        for i in range(len(w_changes)):
            self.W[i] += (self.learning_rate) * w_changes[i]




if __name__ == "__main__":

    # DATA
    p = 2
    inputs, labels = paritaMinus(p)

    threshold = 0
    label = -1

    # PARAMS
    tahn = Tahn()
    quasi = Quasi()

    params = {
        "layers": [p, 5, 5, 3, 1],
        "activation_functions": [tahn, quasi, tahn, tahn],
        "learning_rate": 0.5,
        "weight_mean": 0.0,
        "weight_variance": 1
    }

    # DODO ide ?
    network = NASA(params)
    success_global = 0
    epoch = 0
    succ_max = 0
    success_window = 10
    max_epoch = 1000
    indexer = list(range(len(inputs)))

    while success_global < success_window and epoch < max_epoch:
        # random.shuffle(indexer)
        success_epoch = 0
        for i in indexer:
            intput = np.reshape(inputs[i], (p, 1))
            h = network.activation(intput)
            y = h[-1]

            if y[0][0] >= threshold and labels[i][0] == 1 or y[0][0] < threshold and labels[i][0] == label:
                success_epoch += 1
            network.learning(h, labels[i])

            if success_epoch > succ_max:
                succ_max = success_epoch

        if success_epoch == 2 ** p:
            success_global += 1

        epoch += 1

    print(f"XOR sucess {(success_epoch == 2 ** p)}. Epochs to success: {epoch}. {succ_max} out of {2 ** p}")
