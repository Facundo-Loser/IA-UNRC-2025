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

# obtener el costo de ir de la ciudad A a la B
def get_cost(cityA, cityB):
    return CITIES_MATRIX[cityA][cityB]

# obtener el tiempo de ir de A y B
def get_time(cityA, cityB):
    return CITIES_TIMES[cityA][cityB]

# fitness costo
def fitnessCost(individual):
   cost = 0
   for i in range(CANT_CITIES):
       cost += get_cost(individual[i], individual[i+1])
   return cost

# fitness tiempo
def fitnessTime(individual):
    time = 0
    for i in range(CANT_CITIES):
        time += get_time(individual[i], individual[i+1])
    return time

# no es necesario
def create_individual():
    pass

# saco la ciudad de inicio, luego ahgo todas las permutaciones y le vuelvo a
# concatenar al principio y al final la ciudad de inicio
def create_population():
    #cities = [0, 1, 2, 3]
    #  creo la lista de ciudades
    cities = []
    for i in range(CANT_CITIES):
        cities.append(i)

    cities.remove(START_CITY)
    permutaciones = list(itertools.permutations(cities))
    permutaciones = [list(p) for p in permutaciones]
    return [[START_CITY] + permutaciones[i] + [START_CITY] for i in range(POPULATION_SIZE)]

def selection(population):
    f1 = population.sort(key=fitnessCost)
    f2 = population.sort(key=fitnessTime)
    f1_values = []
    f2_values = []
    f1_plus_f2 = []
    max_f1 = f1[-1]
    min_f1 = f1[0]
    max_f2 = f2[-1]
    min_f2 = f2[0]
    
    for _ in range(CANT_CITIES-1):
        f1_values.append(float('inf'))
        f1_values.append(float('inf'))

    f1_plus_f2.append(float('inf'))

    for i in range(1,CANT_CITIES-1):
        f1_values[i] = abs(f1_values[i-1] - f1_values[i+1])/(max - min)
        f2_values[i] = abs(f2_values[i-1] - f2_values[i+1])/(max - min)
        f1_plus_f2.append(f1_values[i] + f2_values[i])

    f1_plus_f2.append(float('inf'))    

# al hacer crossover excluimos la ciudad de inicio para no crear individuos invalidos
# basicamente tomamos la primer parte de un padre y rellenamos con el otro pero
# sin repretir ciudades para no obbtener un individuo invalido
def crossover(p1, p2):
    # combination of one point
    if random.random() < CROSOVER_PROB:
        # excluyo la ciudad de inicio (y de fin)
        p1_inner = p1[1:-1]
        p2_inner = p2[1:-1]
        point = random.randint(1, len(p1_inner) - 1)
        offspring1 = [START_CITY] + p1_inner[:point]
        offspring2 = [START_CITY] + p2_inner[:point]

        # ahora relleno el resto con el otro padre
        for city in p2_inner:
            if not (city in offspring1):
                offspring1.append(city)

        for city in p1_inner:
            if not (city in offspring2):
                offspring2.append(city)

        offspring1.append(START_CITY)
        offspring2.append(START_CITY)
        return offspring1, offspring2
    return p1[:], p2[:]

# intercambia dos ciudades de lugar menos la ciudad de inicio
def mutate(individual):
    if random.random() < MUTATION_PROB:
            index1 = random.randint(1, len(individual)-2)
            index2 = random.randint(1, len(individual)-2)
            elem1 = individual[index1]
            elem2 = individual[index2]
            individual[index1] = elem2
            individual[index2] = elem1
    return individual

# main algorithm
def genetic_algorithm():
    population = create_population()
    for gen in range(GENERATIONS):
        # evaluate and show the best one
        population.sort(key=fitness, reverse=True)
        print(f"Gen {gen}: Best = {population[0]} Fitness = {fitness(population[0])}")

        # selection
        selected = selection(population)

        # reproduction
        next_gen = []
        for i in range(0, POPULATION_SIZE, 2): # the third parameter is the increment or step
            offspring1, offspring2 = crossover(selected[i], selected[i+1])
            next_gen.append(mutate(offspring1))
            next_gen.append(mutate(offspring2))

        population = next_gen
    return min(population, key=fitness) # return the final result

# execute
best = genetic_algorithm()
best.reverse()
print(f"Best individual found: {best}, Fitness = {fitness(best)}")
