# backpack problem: The goal is to determine the set of items that can be included
# in a limited-capacity backpack to maximize the total value of the items

import random

# we use a list of 1's & 0's were 1 represent that
# the element is inside the backpack & 0 is outside

# so our cromosomes are lists of 1's & 0's and
# also we generate 1 global list that have the weights

# params
POPULATION_SIZE = 20
GENERATIONS = 30
CROSSOVER_PROB = 0.8  # crossover probability
MUTATION_PROB = 0.05  # mutation probability
CROMOSOME_LENGTH = 10 # bits
MAX_WEIGHT = 50       # maximum weight allowed to carry in the bag
WEIGHT_RANGE = 20     # generate weights for the elems between 1 & WEIGHT_RANGE

# we want to maximize the fitness function
def fitness(individual, weights):
    points = 0
    total_weight = 0
    for i in range(CROMOSOME_LENGTH):
        if individual[i]:
            total_weight += weights[i]
            if total_weight < MAX_WEIGHT:
                points += 1
            else:
                points -= 1
    return points

def create_individual():
    return [random.randint(0,1) for _ in range(CROMOSOME_LENGTH)]

def create_weights():
    return [random.randint(1,WEIGHT_RANGE) for _ in range(CROMOSOME_LENGTH)]

def create_population():
    return [create_individual() for _ in range(POPULATION_SIZE)]

def selection_by_tournament(population, weights):
    k = 2
    selected = []
    for _ in range(POPULATION_SIZE):
        aspirants = random.sample(population, k)
        winner = max(aspirants, key=lambda ind: fitness(ind, weights))
        selected.append(winner)
    return selected

# combination of one point
def crossover_one_point(p1, p2):
    if random.random() < CROSSOVER_PROB:
        point = random.randint(1, CROMOSOME_LENGTH - 1)
        return p1[:point] + p2[point:], p2[:point] + p1[point:]
    return p1[:], p2[:]

# mutation by gen invertion (we pass the list with 1's & 0's)
def mutate_gen_invertion(individual):
    for i in range(CROMOSOME_LENGTH):
        if random.random() < MUTATION_PROB:
            individual[i] = 1 - individual[i]
    return individual

# main algorithm
def genetic_algorithm():
    weights = create_weights()
    population = create_population()
    for gen in range(GENERATIONS):
        # show the best
        population.sort(key=lambda ind: fitness(ind, weights), reverse=True)
        print(f"Gen {gen}: Best = (elems) {population[0]} (weights) {weights} Fitness = {fitness(population[0], weights)}")

        # selection
        selected = selection_by_tournament(population, weights)

        # reproduction
        next_gen = []
        for i in range(0, POPULATION_SIZE, 2): # step by 2
            offspring1, offspring2 = crossover_one_point(selected[i], selected[i+1])
            next_gen.append(mutate_gen_invertion(offspring1))
            next_gen.append(mutate_gen_invertion(offspring2))
        population = next_gen
    return max(population, key=lambda ind: fitness(ind, weights)), weights # the best one

best, weigths = genetic_algorithm()
print(f"the best individual: (elems) {best} (weights) {weigths}")
