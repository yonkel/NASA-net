import math
import numpy as np
from net_util import SigmoidNp, Quasi


class NASA:
    def __init__(self, params):

        # read hyperparameters
        self.layers = len(params["layers"]) # [ input dimension, h_1, ... , h_n, output dimension ]
        self.activation_funcions = params["activation_funcions"]
        self.learning_rate = params["learning_rate"]

        # initiate Weights
        self.W = []
        for i in range(self.layers - 1):
            if type(self.activation_funcions) == Quasi:
                self.W.append(np.random.normal(params["weight_mean"], params["weight_variance"],
                                               (params["layers"][i], params["layers"][i + 1])))
            else:  # add bias if not Quasi
                self.W.append(np.random.normal(params["weight_mean"], params["weight_variance"],
                                               (params["layers"][i], params["layers"][i + 1] + 1)))

    def activation(self, act_input):

        h = [act_input]

        for i in range(1, self.layers):
            if type(self.activation_funcions[i]) == Quasi:
                h_i = self.activation_funcions[-1].apply_func(self.W[i - 1], h[i - 1])
            else:
                biased_h_i = np.vstack([act_input, np.ones(len(h[i-i][0]))])
                h_i = self.activation_funcions[-1].apply_func(self.W[i - 1], biased_h_i)

            h.append(h_i)

        return h

    def learning(self, veci):
        pass
        # TODO
