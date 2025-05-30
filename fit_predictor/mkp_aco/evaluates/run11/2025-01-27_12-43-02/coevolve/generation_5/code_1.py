import numpy as np
import numpy as np
from scipy.stats import norm

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    n, m = prize.shape[0], weight.shape[1]
    
    # Genetic Algorithm parameters
    population_size = 100
    generations = 50
    mutation_rate = 0.1
    
    # Initialize population
    population = np.random.rand(population_size, n)
    population = (population > 0.5).astype(int)
    
    # Fitness function
    def fitness(individual):
        total_prize = np.dot(individual, prize)
        total_weight = np.dot(individual, weight)
        return total_prize, total_weight
    
    # Genetic operators
    def crossover(parent1, parent2):
        child = np.random.rand(n)
        child[np.random.choice(n, 2, replace=False)] = parent1[np.random.choice(n, 2, replace=False)]
        child[np.random.choice(n, 2, replace=False)] = parent2[np.random.choice(n, 2, replace=False)]
        return (child > 0.5).astype(int)
    
    def mutate(individual):
        individual[np.random.choice(n)] = 1 - individual[np.random.choice(n)]
        return individual
    
    # Evolution
    for generation in range(generations):
        population_fitness = np.array([fitness(individual) for individual in population])
        population = population[population_fitness[:, 0].argsort()[::-1]]
        
        new_population = []
        while len(new_population) < population_size:
            parent1, parent2 = population[np.random.choice(population_size, 2, replace=False)]
            child = crossover(parent1, parent2)
            if np.random.rand() < mutation_rate:
                child = mutate(child)
            new_population.append(child)
        population = np.array(new_population)
    
    # SLS algorithm
    def stochastic_local_search(individual):
        for _ in range(100):
            neighbor = np.copy(individual)
            index = np.random.randint(n)
            neighbor[index] = 1 - neighbor[index]
            neighbor_fitness, _ = fitness(neighbor)
            if neighbor_fitness > fitness(individual)[0]:
                individual = neighbor
        return individual
    
    best_individual = population_fitness[:, 0].argmax()
    best_solution = stochastic_local_search(population[best_individual])
    
    # Calculate heuristic scores
    heuristic_scores = np.zeros(n)
    for i in range(n):
        heuristic_scores[i] = norm.cdf(-np.log(1 - np.sum(best_solution[:i+1] * prize[:i+1]) / np.sum(best_solution[:i+1] * weight[:i+1])))
    
    return heuristic_scores
