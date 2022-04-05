import math
import numpy as np


class SigmoidNp:
    def __init__(self):
        pass

    def apply_func(self, net):
        return 1.0 / (1.0 + np.exp(-net))

    def apply_derived(self, output):
        return output * (1 - output)


class SoftMaxNp:
    def __init__(self):
        pass

    def apply_func(self, net):
        e_net = np.exp(net - np.max(net))
        e_denom0 = e_net.sum(axis=0, keepdims=True)
        result = e_net / e_denom0
        return result

    def apply_derived(self, output):
        return output * (1 - output)


class SigmoidList:
    def __init__(self):
        pass

    def apply_func(self, net):
        return [1.0 / (1.0 + math.exp(-x)) for x in net]

    def apply_derived(self, output):
        return [x * (1 - x) for x in output]


class SoftMaxList:
    def __init__(self):
        pass

    def apply_func(self, net_list):
        e_max = max(net_list)
        e_net = [math.exp(net - e_max) for net in net_list]
        e_denom0 = sum(e_net)
        result = [e / e_denom0 for e in e_net]
        return result

    def apply_derived(self, output):
        return [x * (1 - x) for x in output]


class OutputProcessor:
    def __init__(self):
        pass

    def make_one_hot(input_array):
        output_array = np.zeros((input_array.size, input_array.max()+1))
        output_array[np.arange(input_array.size),input_array] = 1
        return output_array

class Exp:
    def __init__(self):
        pass

    def apply_func(self, net):
        print("aktivacia nie je implementovana")
        return None

        # result = np.exp(A @ np.log(B))


    def apply_derived(self, output):
        return 1 - output*output

class Exp2:
    def __init__(self):
        pass

        return np.exp(net)

    def apply_derived(self, output):
        if output >= 0:
            if output > 1 :
                return np.NAN
            return - np.log(1-output/2)

        elif output <= -0:
            if output < -1:
                return np.NAN
            return  np.log(1+output/2)

        return output #NaN

class Tahn:
    def __init__(self):
        pass

    def apply_func(self, x):
        return ( np.exp(x) - np.exp(-x) ) / ( np.exp(x) + np.exp(-x) )

    def apply_derived(self, x):
        # Derivation of Hyperbolic Tangent
        return ( 1 + x ) * ( 1 - x ) # 1^2 - tahn^2(x)