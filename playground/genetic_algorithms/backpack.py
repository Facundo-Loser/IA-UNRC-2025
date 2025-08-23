# backpack problem: The goal is to determine the set of items that can be included
# in a limited-capacity backpack to maximize the total value of the items

import random

# we use a list of 1's & 0's were 1 represent that
# the element is inside the backpack & 0 is outside
# so we use 2 lists: 1 indicates the elems inside
# the bag & the other holds the weights of each elem
# [[elems in],[weights]]

# params
POPULATION_SIZE = 20
GENERATIONS = 30
CROSSOVER_PROB = 0.8  # crossover probability
MUTATION_PROB = 0.05  # mutation probability
CROMOSOME_LENGTH = 10 # bits
MAX_WEIGHT = 50       # maximum weight allowed to carry in the bag

def fitness(individual):
    elems, weights = individual # unpack i think it's called
    points = 0
    total_weight = 0
    for i in range(CROMOSOME_LENGTH):
        if elems[i]:
            total_weight += weights[i]
            if total_weight < MAX_WEIGHT:
                points += 1
            else:
                points -= 1
    return points

def create_individual():
    return [[random.randint(0,1) for _ in range(CROMOSOME_LENGTH)], [random.randint(1,50) for _ in range(CROMOSOME_LENGTH)]]

def create_population():
    return [create_individual() for _ in range(POPULATION_SIZE)]

def selection_by_tournament(population):
    k = 2
    selected = []
    for _ in range(POPULATION_SIZE):
        aspirants = random.sample(population, k)
        winner = max(aspirants, key=fitness)
        selected.append(winner)
    return selected

# combination of one point
def crossover_one_point(p1, p2):
    bits1, weights1 = p1
    bits2, weights2 = p2

    if random.random() < CROSSOVER_PROB:
        point = random.randint(1, CROMOSOME_LENGTH - 1)
        new_bits1 = bits1[:point] + bits2[point:]
        new_bits2 = bits2[:point] + bits1[point:]
    else:
        new_bits1 = bits1[:]
        new_bits2 = bits2[:]

    return [new_bits1, weights1[:]], [new_bits2, weights2[:]]

# mutation by gen invertion (we pass the list with 1's & 0's)
def mutate_gen_invertion(individual):
    bits, weights = individual
    for i in range(CROMOSOME_LENGTH):
        if random.random() < MUTATION_PROB:
            bits[i] = 1 - bits[i]
    return [bits, weights[:]]

# main algorithm
def genetic_algorithm():
    population = create_population()
    for gen in range(GENERATIONS):
        # show the best
        population.sort(key=fitness, reverse=True)
        print(f"Gen {gen}: Best = (elems){population[0][0]} (weights) {population[0][1]} Fitness = {fitness(population[0])}")

        # selection
        selected = selection_by_tournament(population)

        # reproduction
        next_gen = []
        for i in range(0, POPULATION_SIZE, 2): # step by 2
            offspring1, offspring2 = crossover_one_point(selected[i], selected[i+1])
            next_gen.append(mutate_gen_invertion(offspring1))
            next_gen.append(mutate_gen_invertion(offspring2))
        population = next_gen
    return max(population, key=fitness) # the best one

best = genetic_algorithm()
print(f"the best individual: (elems) {best[0]} (weights) {best[1]}")
