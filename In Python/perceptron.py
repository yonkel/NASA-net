import numpy as np


class Perceptron:
    def __init__(self, layers, activation_funcions, learning_rate, init_w_mean=0.0, init_w_variance=1.0):
        # just to be sure let's remember the sizes of layers of our network we call it the network architecture
        # it's a simple list, I can imagine a dictionary may be even more readable {input: sth, hidden: ...
        # we assume only 3 layers
        self.arch = layers
        # activation functions are objects inserted when the net is born, they need to have 2 methods: apply_func and apply_derived
        # for 3 layer net there will be 2 items in this list, similarly a dict would maybe do better for legibility of the code
        self.activation_funcions = activation_funcions
        self.learning_rate = learning_rate
        self.init_weight_mean = init_w_mean
        self.init_weight_variance = init_w_variance
        # make weights: notice +1 for bias, we have to keep in mind to add the bias to all computations
        self.weights_input_hidden = np.random.normal(self.init_weight_mean, self.init_weight_variance,
                                                     (self.arch[1], self.arch[0] + 1))
        self.weights_hidden_output = np.random.normal(self.init_weight_mean, self.init_weight_variance,
                                                      (self.arch[2], self.arch[1] + 1))

        self.weights_input_hidden = np.array([[-0.15968122, 0.69476766, -0.64096558],
                                              [1.40581777, -0.6320973, 0.82332406],
                                              [1.24587293, -2.02262976, 0.71242802],
                                              [1.51597793, 0.72899789, 1.08629178],
                                              [-0.55620016, -1.58849003, -0.04311313],
                                              [1.53407221, 0.0981844, -0.65777507]])

        self.weights_hidden_output = np.array(
            [[-1.90690636, 2.32660823, -1.96985584, 1.59555433, 0.88560405, -0.46287536,
              -1.67162011]])




    def activation(self, act_input):
        biased_input = np.vstack([act_input, np.ones(len(act_input[0]))])

        act_hidden = self.activation_funcions[0].apply_func(
            np.dot(self.weights_input_hidden, biased_input)
        )
        biased_act_hidden = np.vstack([act_hidden, np.ones(len(act_hidden[0]))])
        act_output = self.activation_funcions[1].apply_func(
            np.dot(self.weights_hidden_output, biased_act_hidden)
        )

        return act_hidden, act_output

    # BP from ICI slides:
    # w_{jk} += \alpha \delta_{k}h_{j}
    # \delta_{k}=(d_{k} - y_{k}) f'_{k},
    # v_{ij} += \alpha \delta_{j} x_{i}
    # \delta_{j}=(\sum_{k}w_{jk}\delta_{k}) f'_{j},
    def learning(self, act_input, act_hidden, act_output, labels):
        # len(act_input[0]) is the minibatch size
        biased_act_input = np.vstack([act_input, np.ones(len(act_input[0]))])
        biased_act_hidden = np.vstack([act_hidden, np.ones(len(act_hidden[0]))])

        delta_output = (labels - act_output) * self.activation_funcions[1].apply_derived(act_output)

        delta_hidden = np.dot(self.weights_hidden_output[:, :self.arch[1]].transpose(), delta_output) * \
                       self.activation_funcions[0].apply_derived(act_hidden)

        weight_change_output = np.dot(biased_act_hidden, delta_output.transpose()).transpose()
        weight_change_hidden = np.dot(biased_act_input, delta_hidden.transpose()).transpose()

        self.weights_hidden_output += (self.learning_rate) * weight_change_output
        self.weights_input_hidden += (self.learning_rate) * weight_change_hidden


        return True
