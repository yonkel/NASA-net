import math

import numpy as np
from net_util import SigmoidNp


class ExpNet:
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

        self.weights_input_hidden = np.random.normal(self.init_weight_mean, self.init_weight_variance,(self.arch[1],self.arch[0]+1))
        self.weights_hidden_output = np.random.normal(self.init_weight_mean, self.init_weight_variance,(self.arch[2],self.arch[1]))
        # self.weights_hidden_output = np.ones((self.arch[2], self.arch[1]))

        # print(self.weights_hidden_output.shape)
        # input() #sedi (1,8)
        self.sigmoid = SigmoidNp()

    def msupermul_left(self, A, B):
        rows = A.shape[0]
        columns = B.shape[1]
        result = np.ones((rows, columns))
        for i in range(rows):
            for j in range(columns):
                for k in range(A.shape[1]):
                    result[i][j] *= B[k][j] ** A[i][k]
        return result

    def quasiPow(self, base, exp):
        return 1 - exp * ( 1 - base )

    def mquasimul_left(self, A, B):
        sigmo = SigmoidNp()
        rows = A.shape[0]
        columns = B.shape[1]
        result = np.ones((rows, columns))
        for i in range(rows):
            for j in range(columns):
                for k in range(A.shape[1]):
                    result[i][j] *= self.quasiPow( B[k][j] ,sigmo.apply_func(A[i][k]) )
        return result

    def MSE(self, inputs, labels):
        SSE = 0

        for i in range(len(labels)):
            intput = np.reshape(inputs[i], (2, 1))
            act_hidden, act_output = self.activation(intput)
            SSE += ( labels[i][0] - act_output[0] )**2

        return SSE / len(labels)

    def activation(self, act_input):

        biased_input = np.vstack([act_input, np.ones(len(act_input[0]))])
        act_hidden = self.activation_funcions[0].apply_func(np.dot(self.weights_input_hidden, biased_input))


        act_output = self.mquasimul_left(self.weights_hidden_output, act_hidden)


        return act_hidden, act_output

    def learning(self, act_input, act_hidden, act_output, labels):
        biased_act_input = np.vstack([act_input, np.ones(len(act_input[0]))])
        weight_change_output = self.weights_hidden_output.copy()
        error_hid = act_hidden.copy()

        error_out = (labels - act_output)
        delta_output = error_out * act_output

        for j in range(self.arch[1]):
            tmp = 0
            for k in range(self.arch[2]):
                logistic_degree = self.sigmoid.apply_func(self.weights_hidden_output[k][j])

                common_term_j_k = self.quasiPow(act_hidden[j][0], logistic_degree)

                # print("term",common_term_j_k)
                # print("delta[k]",delta_output[k])

                if common_term_j_k < 0.0000000001:
                    # print("j=",j,",k=",k,",base=act_hidden[j][0]=",act_hidden[j][0],",degree=logistic(outputWeight)=",logistic_degree,",deltaOutput[k]=",delta_output[k],",quasiPow=",self.quasiPow(act_hidden[j][0], logistic_degree))
                    # print("error_out[k]=",error_out[k][0],",act_output[k][0]=",act_output[k][0])
                    # print("quasipow=(1-degree*(1-base)):(1-base)=",(1-act_hidden[j][0]),",degree*(1-base)=",(logistic_degree*(1-act_hidden[j][0])))
                    common_term_j_k = 1.0;
                    for j2 in range(self.arch[1]):
                        if j2 != j:
                            common_term_j_k *= self.quasiPow(act_hidden[j2][0], self.sigmoid.apply_func(self.weights_hidden_output[k][j2]))
                    # print("productDerivative[k]=", common_term_j_k)
                    common_term_j_k *= error_out[k][0]
                else:
                    common_term_j_k = delta_output[k] / common_term_j_k

                if (math.isnan(common_term_j_k)):
                    print("j=",j,",k=",k,",base=act_hidden[j][0]=",act_hidden[j][0],",degree=logistic(outputWeight)=",logistic_degree,",deltaOutput[k]=",delta_output[k],",quasiPow=",self.quasiPow(act_hidden[j][0], logistic_degree))
                    print("error_out[k]=",error_out[k][0],",act_output[k][0]=",act_output[k][0])
                    print("quasipow=(1-degree*(1-base)):(1-base)=",(1-act_hidden[j][0]),",degree*(1-base)=",(logistic_degree*(1-act_hidden[j][0])))
                    quit()

                weight_change_output[k][j] = common_term_j_k * (act_hidden[j] - 1) * logistic_degree * (1 - logistic_degree)

                tmp += common_term_j_k * logistic_degree

            error_hid[j] = tmp


        # error_hid = np.dot(self.weights_hidden_output[:, :self.arch[1]].transpose(), delta_output) / act_hidden
        delta_hidden = error_hid * self.activation_funcions[0].apply_derived(act_hidden)


        # weight_change_output = np.dot(act_hidden, delta_output.transpose()).transpose()
        weight_change_hidden = np.dot(biased_act_input, delta_hidden.transpose()).transpose() # a @ b.T == (b @ a.T).T


        self.weights_hidden_output += (self.learning_rate) * weight_change_output
        self.weights_input_hidden += (self.learning_rate) * weight_change_hidden



        # #############################################
        # print("labels\n", labels, "\n")
        # print("act_out\n", act_output, "\n")
        # print("act_input\n", act_input, "\n")
        # print("error_out\n", error_out, "\n")
        # print( "delta_output\n", delta_output, "\n")
        # print( "act_hidden\n", act_hidden, "\n")
        # print("error_hid\n", error_hid, "\n")
        # print("delta_hidden\n", delta_hidden, "\n")
        # print("zmena\n", weight_change_hidden, "\n")
        # #############################################


        return True

from net_util import Exp, Tahn
exp = Exp()
tahn = Tahn()
n = ExpNet([2,3,1], [tahn,exp], 0.5)

act_input = np.reshape( np.array([0,0]), (2,1))

act_output = np.reshape( np.array([-0.45281027404449]), (1,1))

act_hidden = np.reshape( np.array([
0.9906782365800988 ,
-0.7314259368233821 ,
0.6249039835614231 ,
]), (3,1))

label = np.reshape( np.array([0]), (1,1))

n.learning( act_input, act_hidden, act_output, label)

''' learning

target = label
0.0

result.output = act_output == result.state.activation[i]
-0.45281027404449

result.state.activation[i-2] == act_input 
0.0 
0.0 

result.errorSignal[i] == error_out
0.45281027404449 

result.delta[i] == delta_out
-0.2050371442802461 

result.state.activation[i-1] = act_hidden
0.9906782365800988 
-0.7314259368233821 
0.6249039835614231 

result.errorSignal[i-1] == error_hid
-0.2069664364365678 
0.2803252304269278 
-0.328109837149234 

result.delta[i-1] == delta_hid 
-0.0038405999075943205 
0.13035574508815784 
-0.19998130891044993 

result.deltaWeight[i-2] == zmena 
0.0 0.0 -0.0038405999075943205 
0.0 0.0 0.13035574508815784 
0.0 0.0 -0.19998130891044993 



'''







''' propagation 
weights[layer]
2.726795083003882 0.7615135982311503 0.988217528686029 
2.2420442598488437 -1.4929417166690235 2.4539262179673704 
0.8710175789190617 0.5619429562832592 0.16001984371656253 

state.activation[layer]
0.0 
0.0 

state.activation[layer + 1]
0.7566012477015271 
0.9853316979059518 
0.15866784849975385 

state.activation[layer + 2]
0.11828738752997277 

#################################
w_hid = np.array(
[[2.726795083003882, 0.7615135982311503, 0.988217528686029],
[2.2420442598488437, -1.4929417166690235, 2.4539262179673704  ],
[0.8710175789190617, 0.5619429562832592, 0.16001984371656253 ]
    ])
print("w_hid", w_hid.shape ,"\n",w_hid, "\n")
n.weights_input_hidden = w_hid

input = np.reshape( np.array([0,0]), (2,1))
print("input", input.shape,"\n", input, "\n")

h, y = n.activation(input)

print("act_hidden", h.shape)
print(h, "\n")
print("y\n", y, "\n")



'''


