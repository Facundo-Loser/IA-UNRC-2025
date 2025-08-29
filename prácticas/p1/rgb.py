import random

# this problem consist on: given the decimal representation
# of a number fiond the binary representation

POPULATION_SIZE = 20
GENERATIONS = 30
CROSOVER_PROB = 0.8
MUTATION_PROB = 0.04
GENOME_LENGTH = 3
R = 209  # 255 is the max value
G = 255
B = 207

# fitness function
def fitness(individual):
    #return (R - individual[0])**2 + (G - individual[1])**2 + (B - individual[2])**2
    return (abs(R -  individual[0]) + abs(G - individual[1]) + abs(B - individual[2]))/3

def create_individual():
    return [random.randint(0,255) for _ in range(GENOME_LENGTH)]

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
            individual[i] = random.randint(0, 255)
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
