from deap import base, creator, tools, algorithms
import random

# it uses moga & nsga-2

# 1 define the fitness type & the individual (multiobjective, minimization)
creator.create("FitnessMin", base.Fitness, weights=(-1.0,-1.0)) # minimize both objectives
creator.create("Individual", list, fitness=creator.FitnessMin)

# 2 toobox: generate individuals & population
toolbox = base.Toolbox()
toolbox.register("attr_float", random.uniform, 0, 10)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, 2)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# 3 eval funcion: cost vs contamination
def eval_cost_pollution(individual):
    x, y = individual
    cost = x**2 + y**2               # costo cuadratico center (0,0)
    pollution = (x-5)**2 + (y-5)**2  # pollution center in (5,5)
    return cost, pollution

toolbox.register("evaluate", eval_cost_pollution)

# 4 genetic operators
toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=0, up=10, eta=20.0)
toolbox.register("mutate", tools.mutPolynomialBounded, low=0, up=10, eta=20.0, indpb=0.2)
toolbox.register("select", tools.selNSGA2)

# 5 main algorithm
def main():
    random.seed(42)
    population = toolbox.population(n=50)
    hof = tools.ParetoFront()
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", lambda fits: tuple(sum(f)/len(f) for f in zip(*fits)))
    stats.register("min", lambda fits: tuple(min(f) for f in zip(*fits)))
    stats.register("max", lambda fits: tuple(max(f) for f in zip(*fits)))

    # Evolucionar con NSGA-II
    algorithms.eaMuPlusLambda(population, toolbox, mu=50, lambda_=100, cxpb=0.7, mutpb=0.3,
    ngen=40, stats=stats, halloffame=hof, verbose=True)
    print("\n--- Frente de Pareto ---")
    for ind in hof:
        print(ind, ind.fitness.values)

main()