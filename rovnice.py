# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import math

import numpy as np
import random


# povedzme teoreticky ze mame iba jednu skrytu vrstvu
# vahy W_hid, W_out
# vstup nejaky vektor X



# ACTIVATIOOOOOON
#
# for (layer=0; layer < this.weights.length; layer += 2) {
#     state.activation[layer + 1] = weights[layer].mmul(withBias(state.activation[layer])).apply(Math::tanh);
#     state.activation[layer + 2] = weights[layer + 1].msupermul_leftExponent(state.activation[layer + 1]);


# mmul = DOT PRODUCT

# prva je tradicna -->  Tahn ( W_hid @ ( x + bias ) )

# druha je ta speci to skusim nakodit -->  z[k]  =  exp ( Q + d[k] ) ; Q = sum[for all j] W_out[j][k] * log(y[j])
# ale ta tu nie je ako je tu ta prva  z_k = product[for all j](y_j^D_{j,k} * d_k)

#     public MatrixOfDouble msupermul_leftExponent(MatrixOfDouble that) {
#         guardMatrixMultiplicationDimensions(that);
#         MatrixOfDouble result = new MatrixOfDouble(this.rows, that.columns);
#         for(int i = 0; i < this.rows; i++)
#             for(int j = 0; j < that.columns; j++) {
#                 result.matrix[i][j] = 1.0;
#                 for (int k = 0; k < this.columns; k++)
#                     result.matrix[i][j] *= Math.pow(that.matrix[k][j], this.matrix[i][k]);
#             }
#         return result;
#     }

# The java.lang.Math.exp(double a) returns Euler's number e raised to the power of a double value.
# this = A ---> W vahy alebo v rovniciach D
# that = B ---> hodnoty predoslej


def msupermul_left (A, B):
    if kontrola(A,B) == False:
        return

    rows = A.shape[0]
    columns = B.shape[1]
    result = np.ones((rows, columns))
    for i in range(rows):
        for j in range(columns):
            for k in range(A.shape[1]):
                result[i][j] *= B[k][j] ** A[i][k]
    return result

def left( A, B ): #left supermultiplication A**B is B^A = exp(A*ln(B))
    if kontrola(A,B) == False:
        return

    result = np.exp(A @ np.log(B))
    return result

def right( A, B ): #right supermultiplication A**B is A^B = exp(ln(A)*B)
    if kontrola(A,B) == False:
        return

    result = np.exp(np.log(A) @ B)
    return result

def kontrola(A, B ):
    if A.shape[1] != B.shape[0]:
        print("zle rozmery", A.shape, B.shape)
        return False
    return True



#   z_k = exp(sum[for all j]( D_{j,k} * log(y_j) ) + d_k)


# sum[for all j]( D_{j, k} * log(y_j) )
#      for i in range(rows):
#         for j in range(columns):
#             for k in range(A.size[1]): --> sum[for all j]
#               result[i][j] += D[i][k] * log( Y[k][j] )
#  a to teoreticky mozem spravit ako D @ log( Y ) ??????????



if __name__ == '__main__':

    W_ = [ 1.0, 1.0, 1.0 ]
    h_ = [
        0.6791076844250947,
        - 0.9958341879728652,
        0.26359534306964494
    ]
    y = [0.7239687310440837 ]

    W = np.reshape( np.array(W_), (1,3))
    h = np.reshape( np.array(h_), (3,1))
    # print(W, '\n', W.shape ,'\n')
    # print(h, '\n',h.shape, '\n',)

    # ll = left(W, h) #np.exp(A @ np.log(B))
    # print(ll, ll.shape)

    # rr = right(W, h) #np.exp(np.log(A) @ B)
    # print(rr, rr.shape)

    print(left(W,h))
    print(msupermul_left(W,h))
    print(right(W,h))

    h2_ = [
    0.8438934893172371,
    0.9974878639235529,
    0.8600516872658023,
    ]




