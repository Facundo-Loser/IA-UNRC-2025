# backpack problem: The goal is to determine the set of items that can be included
# in a limited-capacity backpack to maximize the total value of the items

# we have the same problem but now we solve it using deep lib

from deap import base, creator, tools, algorithms
import random

CROMOSOME_LENGTH = 10 # bits
MAX_WEIGHT = 50       # maximum weight allowed to carry in the bag

weigths = [random.randint(1,20) for _ in range(CROMOSOME_LENGTH)]

# 1 create fitness and individual class
creator.create("FitnessMax", base.Fitness, weights=(1.0,))     # maximize fitness
creator.create("Individual", list, fitness=creator.FitnessMax) # ...

# 2 toolbox: attributes, individuals, population
toolbox = base.Toolbox()

# binary atribute
toolbox.register("attr_bool", random.randint, 0, 1)
#individual (each one has 10 bits)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, CROMOSOME_LENGTH)
# population
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# 3 eval function
def eval_backpack(individual):
    total_weight = 0
    points = 0
    for i in range(CROMOSOME_LENGTH):
        if individual[i]:
            total_weight += weigths[i]
            if total_weight < MAX_WEIGHT:
                points += 1
            else:
                points -= 2
    return points,
toolbox.register("evaluate", eval_backpack)

# 4 genetic operators
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
toolbox.register("select", tools.selTournament, tournsize=3)

# 5 main algorithm
def main():
    random.seed(42)
    population = toolbox.population(n=50) # initial population
    algorithms.eaSimple(population, toolbox, cxpb=0.5, mutpb=0.2, ngen=20, verbose=True)

main()