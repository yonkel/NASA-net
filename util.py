
def save_results( net_name, exp_name, nets, epc):
    with open(f'results/{net_name}_{exp_name}_nets.txt', 'w') as f:
        f.write('x y\n')
        f.writelines(nets)
    with open(f'results/{net_name}_{exp_name}_epcs.txt', 'w') as f:
        f.write('x y err\n')
        f.writelines(epc)


if __name__ == '__main__':
    pole = []
    print("Zadaj pocty na skrytej vrstve, ukoncis x")
    while ( True  ):
        try:
            x = input()
            pole.append(int(x))
        except:
            break

    print(pole)
