import numpy as np
import random
from deap import base, creator, tools, algorithms
from skfuzzy import control as ctrl
from fis_controller import pendulum_control

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("attr_float", random.uniform, -50, 50)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=4)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)


def evalPendulumControl(individual):
    angle, angular_velocity = individual[0], individual[1]
    pendulum_simulator = ctrl.ControlSystemSimulation(pendulum_control)

    pendulum_simulator.input['angle'] = angle
    pendulum_simulator.input['angular_velocity'] = angular_velocity

    pendulum_simulator.compute()

    action = pendulum_simulator.output['action']

    score = max(0, 100 - abs(angle)) + max(0, 100 - abs(angular_velocity))

    return (score,)


toolbox.register("evaluate", evalPendulumControl)
toolbox.register("mate", tools.cxBlend, alpha=0.5)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=3)


def optimize_fuzzy_rules():
    pop = toolbox.population(n=50)
    hof = tools.HallOfFame(1)

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("min", np.min)
    stats.register("max", np.max)

    algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2, ngen=40, stats=stats, halloffame=hof)

    return hof[0]


if __name__ == '__main__':
    best_individual = optimize_fuzzy_rules()
    print("Melhor indivíduo:", best_individual)
