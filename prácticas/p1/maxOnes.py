import random

# This problem consists of finding the chain with the max ammount of 1's

# Params
POP_SIZE = 20      # population size
GENS = 30          # number of generations
CXPB = 0.8         # probability of crossing
MUTPB = 0.05       # probability of mutation
GENOME_LENGTH = 20 # length of each individual (number of bits)

# Fitness function
def fitness(individual):
    return sum(individual) # sum: calculates the sum of the elements of an iterable

# create population
def create_individual():
    return [random.randint(0,1) for _ in range(GENOME_LENGTH)]

def create_population():
    return [create_individual() for _ in range(POP_SIZE)]

# genetics operators
def selection(population):
    # selection by tournament
    k = 3
    selected = []
    for _ in range(POP_SIZE):
        # random.sample() is a function of Python's random module that returns a
        # new list with k unique elements randomly selected without replacement
        # from the sequence population
        aspirants = random.sample(population, k)
        winner = max(aspirants, key=fitness)
        selected.append(winner)
    return selected

def crossover(p1, p2):
    # combination of one point
    if random.random() < CXPB: # random.random() returns a floating point number in [0.0, 1.0)
        point = random.randint(1, GENOME_LENGTH-1)
        return p1[:point] + p2[point:], p2[:point] + p1[point:] # list slices to obtain sublists
    return p1[:], p2[:] # creates a new list containing all the elements of the original list

def mutate(individual):
    for i in range(GENOME_LENGTH):
        if random.random() < MUTPB:
            individual[i] = 1 - individual[i] # mutate the gen (inverse)
    return individual

# main algorithm
def genetic_algorithm():
    population = create_population()
    for gen in range(GENS):
        # evaluate and show the best one
        population.sort(key=fitness, reverse=True)
        print(f"Gen {gen}: Best = {population[0]} Fitness = {fitness(population[0])}")

        # selection
        selected = selection(population)

        # reproduction
        next_gen = []
        for i in range(0, POP_SIZE, 2): # the third parameter is the increment or step
            offspring1, offspring2 = crossover(selected[i], selected[i+1])
            next_gen.append(mutate(offspring1))
            next_gen.append(mutate(offspring2))

        population = next_gen
    return max(population, key=fitness) # return the final result

# execute
best = genetic_algorithm()
print(f"Best individual found: {best}, Fitness = {fitness(best)}")