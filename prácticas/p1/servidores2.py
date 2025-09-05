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

# we have 4 nodes

# tendríamos una lista con informacion, luego cada individuo me dice cual edge
# esta encendido y cual apagado

# edge(A-B) - capacidad mbps - costo impl
edges = [
    #(0, 0, 0, 0),
    (0, 1, 10, 2),
    (0, 2, 20, 6),
    (0, 3, 30, 10),

    (1, 0, 16, 8),
    #(1, 1, 0, 0),
    (1, 2, 8, 5),
    (1, 3, 22, 12),

    (2, 0, 80, 30),
    (2, 1, 20, 16),
    #(2, 2, 0, 0),
    (2, 3, 18, 10)

    (3, 0, 7, 4),
    (3, 1, 1, 2),
    (3, 2, 50, 90),
    #(3, 3, 0, 0),
]

def todos_conectados(individual):
    pass

def get_capacidad(individual):
    pass

def get_costo(individual):
    pass

def evaluate(individual):
    return get_capacidad(individual), get_costo(individual)

def crossover(individual):
    pass

def mutate(individual):
    pass

# 1. Create fitness & individual class
creator.create("FitnessMin", base.Fitness, weights=(-1.0,-1.0))
creator.create("Individual", list, fitness=creator.FitnessMin)