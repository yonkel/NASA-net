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


def paritaMinus(n):
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

if __name__ == "__main__":
    inp, lab = paritaMinus(3)
    #
    for i in range(len(inp)):
        print( inp[i], lab[i]  )


    # print(inp)
    # print(lab)

