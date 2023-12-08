from genetic_fuzzy_controller import optimize_fuzzy_rules


def test():
    best_individual = optimize_fuzzy_rules()
    print("Melhor indivíduo otimizado:", best_individual)


if __name__ == '__main__':
    test()