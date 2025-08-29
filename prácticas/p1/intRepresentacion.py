import random

# this problem consist on: given the decimal representation 
# of a number fiond the binary representation

POPULATION_SIZE = 20
GENERATIONS = 20
CROSOVER_PROB = 0.8
MUTATION_PROB = 0.04
GENOME_LENGTH = 8 # -256 to 255
NUMBER = 8

# fitness function
def fitness(individual):
    dec_ind = 0 # representacion en decimal del individuo
    # hay que recorrer al reves la lista para calcularlo bien
    individual.reverse()
    for i in range(GENOME_LENGTH):
        if (individual[i]):
            dec_ind += 2**i   
    return abs(NUMBER - dec_ind) # return the absolute diference

def create_individual():
    return [random.randint(0,1) for _ in range(GENOME_LENGTH)]

def create_population():
    return [create_individual() for _ in range(POPULATION_SIZE)]

def selection(population):
    # selection by tournament
    k = 3
    selected = []
    for _ in range(POPULATION_SIZE):
        # random.sample() is a function of Python's random module that returns a
        # new list with k unique elements randomly selected without replacement
        # from the sequence population
        aspirants = random.sample(population, k)
        winner = min(aspirants, key=fitness)
        selected.append(winner)
    return selected

def crossover(p1, p2):
    # combination of one point
    if random.random() < CROSOVER_PROB: # random.random() returns a floating point number in [0.0, 1.0)
        point = random.randint(1, GENOME_LENGTH-1)
        return p1[:point] + p2[point:], p2[:point] + p1[point:] # list slices to obtain sublists
    return p1[:], p2[:] # creates a new list containing all the elements of the original list

def mutate(individual):
    for i in range(GENOME_LENGTH):
        if random.random() < MUTATION_PROB:
            individual[i] = 1 - individual[i] # mutate the gen (inverse)
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
