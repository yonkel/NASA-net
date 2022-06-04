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
        self.weights_input_hidden = np.random.normal(self.init_weight_mean, self.init_weight_variance,(self.arch[1],self.arch[0]+1))
        self.weights_hidden_output = np.random.normal(self.init_weight_mean, self.init_weight_variance,(self.arch[2],self.arch[1]+1))

    def MSE(self, inputs, labels):
        threshold = 0.5
        lower_label = 0
        for item in labels:
            if item[0] == -1:
                threshold = 0
                lower_label = -1
                break
            if item[0] == 0:
                break

        SSE = 0
        properly_determined = 0
        for i in range(len(labels)):
            intput = np.reshape(inputs[i], (2, 1))
            act_hidden, act_output = self.activation(intput)
            SSE += (labels[i][0] - act_output[0]) ** 2

            if act_output[0][0] >= threshold and labels[i][0] == 1 or act_output[0][0] < threshold and labels[i][
                0] == lower_label:
                properly_determined += 1

        return (SSE / len(labels), properly_determined)

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


