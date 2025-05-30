import numpy as np
import numpy as np
import random

def select_parent(population):
    fitness_sum = sum(individual['fitness'] for individual in population)
    total = random.uniform(0, fitness_sum)
    current_sum = 0
    for individual in population:
        current_sum += individual['fitness']
        if current_sum > total:
            return individual

def crossover(parent1, parent2):
    child = []
    for gene1, gene2 in zip(parent1, parent2):
        if random.random() < 0.5:
            child.append(gene1)
        else:
            child.append(gene2)
    return child

def mutate(child):
    for i in range(len(child)):
        if random.random() < 0.1:
            child[i] = 1 - child[i]
    return child

def genetic_algorithm(prize, weight):
    population_size = 100
    generations = 100
    mutation_rate = 0.1
    crossover_rate = 0.9
    
    population = [{'chromosome': [random.choice([0, 1]) for _ in range(prize.size)],
                   'fitness': 0} for _ in range(population_size)]
    
    for individual in population:
        individual['fitness'] = sum(prize[i] if weight[i, 0] <= 1 else 0 for i in range(prize.size) if individual['chromosome'][i])
    
    for _ in range(generations):
        new_population = []
        for _ in range(population_size // 2):
            parent1 = select_parent(population)
            parent2 = select_parent(population)
            child = crossover(parent1['chromosome'], parent2['chromosome'])
            child = mutate(child)
            child = np.array(child)
            new_population.append({'chromosome': child, 'fitness': 0})
        
        for individual in new_population:
            individual['fitness'] = sum(prize[i] if weight[i, 0] <= 1 else 0 for i in range(prize.size) if individual['chromosome'][i])
        
        population = new_population
    
    best_individual = max(population, key=lambda x: x['fitness'])
    return best_individual['chromosome']

def stochastic_local_search(best_individual):
    current_individual = best_individual.copy()
    for _ in range(1000):
        random_index = random.randint(0, current_individual.size - 1)
        if random.random() < 0.5:
            current_individual[random_index] = 1 - current_individual[random_index]
        new_fitness = sum(prize[i] if weight[i, 0] <= 1 else 0 for i in range(prize.size) if current_individual[i])
        if new_fitness > current_individual['fitness']:
            current_individual['fitness'] = new_fitness
    
    return current_individual

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    best_individual = genetic_algorithm(prize, weight)
    optimized_individual = stochastic_local_search(best_individual)
    return optimized_individual
