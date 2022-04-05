def parita(n):
    inputs, labels = [],[]

    inputs = [list(bin(x)[2:].rjust(n, '0')) for x in range(2 ** n)]
    for i in range(len(inputs)):
        inputs[i] = list(map(int, inputs[i]))
        if inputs[i].count(1) % 2 == 0 :
            labels.append([0])
        else:
            labels.append([1])

    return inputs, labels


def paritaJedna(n):
    inputs, labels = [],[]

    inputs = [list(bin(x)[2:].rjust(n, '0')) for x in range(2 ** n)]
    for i in range(len(inputs)):
        inputs[i] = list(map(int, inputs[i]))
        if inputs[i].count(1) % 2 == 0 :
            labels.append([-1])
        else:
            labels.append([1])

        for j in range(len(inputs[i])):
            if inputs[i][j] == 0:
                inputs[i][j] = -1

    return inputs, labels


inp, lab = paritaJedna(3)
#
for i in range(len(inp)):
    print( inp[i], lab[i]  )


# print(inp)
# print(lab)

'''
[0, 0, 0] [-1]
[0, 0, 1] [1]
[0, 1, 0] [1]
[0, 1, 1] [-1]
[1, 0, 0] [1]
[1, 0, 1] [-1]
[1, 1, 0] [-1]
[1, 1, 1] [1]
'''