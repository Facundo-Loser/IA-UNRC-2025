from deap import base, creator, tools, algorithms
import random

# 1. Create fitness & individual class
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

# 2 toolbox, attributes, individuals, population
toolbox = base.Toolbox()

# binary attribute (0, 1)
toolbox.register("attr_bool", random.randint, 0, 1)

# an individual is a list of 20 bits
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, 20)

# a population is a list of individuals
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# 3 evaluation function to count 1's
def eval_maxones(individual):
    return sum(individual),
toolbox.register("evaluate", eval_maxones)

# 4 genetic operators
toolbox.register("mate", tools.cxTwoPoint)                   # 2 point combination
toolbox.register("mutate", tools.mutFlipBit, indpb = 0.05)   # mutation bit flip
toolbox.register("select", tools.selTournament, tournsize=3) # selection by tournament

# 5 main algorithm
def main():
    random.seed(42)
    pop = toolbox.population(n=50) # initial population
    algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2, ngen=20, verbose=True)

    best = tools.selBest(pop, k=1)[0]
    print(f"Best individual: {best}")

main()
