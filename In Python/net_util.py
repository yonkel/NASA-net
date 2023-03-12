import math
import numpy as np
from decimal import Decimal


class SigmoidNp:
    def __init__(self):
        pass

    def __repr__(self):
        return "Sigmoid"


    def apply_func(self, net):
        return 1.0 / (1.0 + np.exp(-net))

    def apply_derived(self, output):
        return output * (1 - output)


class SoftMaxNp:
    def __init__(self):
        pass

    def __repr__(self):
        return "SoftMax"

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
        output_array = np.zeros((input_array.size, input_array.max() + 1))
        output_array[np.arange(input_array.size), input_array] = 1
        return output_array


class Exp:
    def __init__(self):
        pass

    def apply_func(self, net):
        print("aktivacia nie je implementovana")
        return None

        # result = np.exp(A @ np.log(B))

    def apply_derived(self, output):
        return 1 - output * output


class Exp2:
    def __init__(self):
        pass

    def apply_derived(self, output):
        if output >= 0:
            if output > 1:
                return np.NAN
            return - np.log(1 - output / 2)

        elif output <= -0:
            if output < -1:
                return np.NAN
            return np.log(1 + output / 2)

        return output  # NaN


class Tahn:
    def __init__(self):
        pass

    def __repr__(self):
        return "Tahn"

    def apply_func(self, x):
        flag = False
        for i in range(len(x)):
            if x[i][0] > 709 or x[i][0] < -709:
                x = np.array([[Decimal(el[0])] for el in x], dtype=object)
                flag = True
                break

        term = (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

        if flag:
            term = np.asarray(term, dtype=float)

        return term

    def apply_derived(self, x):
        # Derivation of Hyperbolic Tangent
        return (1 + x) * (1 - x)  # 1^2 - tahn^2(x)


class Quasi:
    def __init__(self):
        self.sigmoid = SigmoidNp()

    def __repr__(self):
        return "Quasi"

    def quasiPow(self, base, exp):
        return 1 - exp * (1 - base)

    def apply_func(self, A, B):

        rows = A.shape[0]
        columns = B.shape[1]
        result = np.ones((rows, columns))
        for i in range(rows):
            for j in range(columns):
                for k in range(A.shape[1]):
                    result[i][j] *= self.quasiPow(B[k][j], self.sigmoid.apply_func(A[i][k]))

        return result
