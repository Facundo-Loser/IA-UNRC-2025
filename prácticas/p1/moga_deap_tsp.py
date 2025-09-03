from deap import base, creator, tools, algorithms
import random
import itertools

POPULATION_SIZE = 6 # el limite es la cant de permutaciones
GENERATIONS = 30
CROSOVER_PROB = 0.8
MUTATION_PROB = 0.04

# al final cada individuo termina teniendo 5 ciudades ya que
# se le vuelve a concatenar la ciudad de la que se parte
CANT_CITIES = 4
START_CITY = 0 # ciudad de la que se parte

# hay que hacer una amtriz apra guardar los costos de las aristas del grafo
CITIES_MATRIX = [
#   0   1   2   3
   [0,  10, 40, 1 ], # 0
   [10, 0,  4,  12], # 1
   [40, 4,  0,  6 ], # 2
   [1,  12, 6,  0 ]  # 3
]

CITIES_TIMES = [
#    0  1   2  3
    [0, 20, 4, 12], # 0
    [20, 0, 10, 6], # 1
    [4, 10, 0, 16], # 2
    [12, 6, 16, 0]  # 3
]

from deap import base, creator, tools, algorithms
import random

# 1. Create fitness & individual class
creator.create("FitnessMin", base.Fitness, weights=(-1.0,-1.0))
creator.create("Individual", list, fitness=creator.FitnessMin)

# 2 toolbox, attributes, individuals, population
toolbox = base.Toolbox()

def create_population():
    cities = list(range(CANT_CITIES))
    cities.remove(START_CITY)
    permutaciones = list(itertools.permutations(cities))
    permutaciones = [list(p) for p in permutaciones]

    poblacion = []
    for i in range(POPULATION_SIZE):
        ruta = [START_CITY] + permutaciones[i] + [START_CITY]
        poblacion.append(creator.Individual(ruta))
    return poblacion

# a population is a list of individuals
toolbox.register("population", create_population)

# obtener el costo de ir de la ciudad A a la B
def get_cost(cityA, cityB):
    return CITIES_MATRIX[cityA][cityB]

# obtener el tiempo de ir de A y B
def get_time(cityA, cityB):
    return CITIES_TIMES[cityA][cityB]

# fitness costo
def evalCost(individual):
   cost = 0
   for i in range(CANT_CITIES):
       cost += get_cost(individual[i], individual[i+1])
   return cost

# fitness tiempo
def evalTime(individual):
    time = 0
    for i in range(CANT_CITIES):
        time += get_time(individual[i], individual[i+1])
    return time

def eval(individual):
    return evalCost(individual), evalTime(individual)

toolbox.register("evaluate", eval)

def crossover(p1, p2):
    if random.random() < CROSOVER_PROB:
        p1_inner = p1[1:-1]
        p2_inner = p2[1:-1]
        point = random.randint(1, len(p1_inner) - 1)
        offspring1 = [START_CITY] + p1_inner[:point]
        offspring2 = [START_CITY] + p2_inner[:point]

        for city in p2_inner:
            if city not in offspring1:
                offspring1.append(city)
        for city in p1_inner:
            if city not in offspring2:
                offspring2.append(city)

        offspring1.append(START_CITY)
        offspring2.append(START_CITY)

        return creator.Individual(offspring1), creator.Individual(offspring2)
    return creator.Individual(p1[:]), creator.Individual(p2[:])


# 4 genetic operators
toolbox.register("mate", crossover)                   # 2 point crosover

def mutate(individual):
    if random.random() < MUTATION_PROB:
        index1 = random.randint(1, len(individual)-2)
        index2 = random.randint(1, len(individual)-2)
        individual[index1], individual[index2] = individual[index2], individual[index1]
    return individual,

toolbox.register("mutate", mutate)   # mutation bit flip

toolbox.register("select", tools.selNSGA2)

# 5 main algorithm
def main():
    random.seed(42)
    pop = toolbox.population() # initial population
    hof = tools.ParetoFront()
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", lambda fits: tuple(sum(f)/len(f) for f in zip(*fits)))
    stats.register("min", lambda fits: tuple(min(f) for f in zip(*fits)))
    stats.register("max", lambda fits: tuple(max(f) for f in zip(*fits)))

    # evol con NSGA-II
    algorithms.eaMuPlusLambda(pop, toolbox, mu=50, lambda_=100, cxpb=0.7, mutpb=0.3, ngen=40, stats=stats, halloffame=hof, verbose=True)
    print("\n--- Frente de Pareto ---")
    for ind in hof:
        print(ind, ind.fitness.values)

main()
