from fis_controller import control_pendulum


def test():
    print("Teste 1: ", control_pendulum(0, 0))
    print("Teste 2: ", control_pendulum(10, -5))


if __name__ == '__main__':
    test()