import numpy as np


class ExpNet:
    def __init__(self, layers, activation_funcions, learning_rate, init_w_mean=0.0, init_w_variance=0.02):
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
        # self.weights_hidden_output = np.random.normal(self.init_weight_mean, self.init_weight_variance,(self.arch[2],self.arch[1]))
        self.weights_hidden_output = np.ones((self.arch[2], self.arch[1]))

        # print(self.weights_hidden_output.shape)
        # input() #sedi (1,8)

    def left(self, A, B):  # left supermultiplication A**B is B^A = exp(A*ln(B))
        result = np.exp(A @ np.log(B))
        return result

    def msupermul_left(self, A, B):

        rows = A.shape[0]
        columns = B.shape[1]
        result = np.ones((rows, columns))
        for i in range(rows):
            for j in range(columns):
                for k in range(A.shape[1]):
                    result[i][j] *= B[k][j] ** A[i][k]
        return result

    def activation(self, act_input):

        biased_input = np.vstack([act_input, np.ones(len(act_input[0]))])
        act_hidden = self.activation_funcions[0].apply_func(np.dot(self.weights_input_hidden, biased_input))

        # print("self.weights_hidden_output ",self.weights_hidden_output.shape)
        # print(self.weights_hidden_output, "\n")
        #
        # print("act_hidden", act_hidden.shape)
        # print(act_hidden)


        net_output = self.msupermul_left(self.weights_hidden_output, act_hidden)



        # print("net_output", net_output.shape)
        # print(net_output)
        # input()

        act_output = self.activation_funcions[1].apply_func( net_output )

        # print("act_output", act_output.shape)
        # print(act_output)
        #
        # input()

        return act_hidden, act_output

    # BP from ICI slides:
    # w_{jk} += \alpha \delta_{k}h_{j}
    # \delta_{k}=(d_{k} - y_{k}) f'_{k},
    # v_{ij} += \alpha \delta_{j} x_{i}
    # \delta_{j}=(\sum_{k}w_{jk}\delta_{k}) f'_{j},
    def learning(self, act_input, act_hidden, act_output, labels):

        biased_act_input = np.vstack([act_input, np.ones(len(act_input[0]))])
        delta_output = (labels - act_output) * act_output

        delta_hidden = np.dot(self.weights_hidden_output[:, :self.arch[1]].transpose(), delta_output) * \
                       self.activation_funcions[0].apply_derived(act_hidden)

        # weight_change_output = np.dot(act_hidden, delta_output.transpose()).transpose()
        weight_change_hidden = np.dot(biased_act_input, delta_hidden.transpose()).transpose()


        # self.weights_hidden_output += (self.learning_rate) * weight_change_output
        self.weights_input_hidden += (self.learning_rate) * weight_change_hidden
        return True


