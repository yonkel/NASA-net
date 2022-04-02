def generuj(n):
    inputs, labels = [],[]

    for i in range(n+1):
        [ list(bin(x)[2:].rjust(n, '0')) for x in range(2 ** n)]

    inputs = [list(bin(x)[2:].rjust(n, '0')) for x in range(2 ** n)]
    for i in range(len(inputs)):
        inputs[i] = list(map(int, inputs[i]))
        labels.append(inputs[i].count(1) % 2 )

    return inputs, labels

inp, lab = generuj(10)

for i in range(len(inp)):
    print( inp[i], lab[i]  )

