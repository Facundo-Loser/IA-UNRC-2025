from deap import base, creator, tools, algorithms
import random

POPULATION_SIZE = 20
GENERATIONS = 20
CROSOVER_PROB = 0.8
MUTATION_PROB = 0.04
GENOME_LENGTH = 8 # -256 to 255
NUMBER = 8

# create fitness & individual class
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

# 2 toolbox, attributes, individuals, population
toolbox = base.Toolbox()

# binary attribute (0, 1)
toolbox.register("attr_bool", random.randint, 0, 1)

# an individual is a list of 20 bits
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, GENOME_LENGTH)

# a population is a list of individuals
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# 3 evaluation function to count 1's
def eval_binary(individual):
    dec_ind = 0 # representacion en decimal del individuo
    # hay que recorrer al reves la lista para calcularlo bien
    individual.reverse()
    for i in range(GENOME_LENGTH):
        if (individual[i]):
            dec_ind += 2**i   
    return abs(NUMBER - dec_ind),
toolbox.register("evaluate", eval_binary)

# 4 genetic operators
toolbox.register("mate", tools.cxTwoPoint)                   # 2 point crosover
toolbox.register("mutate", tools.mutFlipBit, indpb = 0.05)   # mutation bit flip
toolbox.register("select", tools.selTournament, tournsize=3) # selection by tournament

# 5 main algorithm
def main():
    random.seed(42)
    pop = toolbox.population(n=20) # initial population
    algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2, ngen=20, verbose=True)

    best = tools.selBest(pop, k=1)[0]
    print(f"Best individual: {best}")

main()