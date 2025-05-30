import numpy as np
import numpy as np

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    n, m = prize.shape[0], weight.shape[1]
    # Initialize the heuristic array with zeros
    heuristics = np.zeros(n)
    
    # Genetic Algorithm Components
    population_size = 100
    mutation_rate = 0.1
    elite_size = 10
    generations = 50
    
    # Stochastic Local Search Components
    local_search_iterations = 10
    temperature = 1.0
    
    # Reinforcement Learning Components
    alpha = 0.1  # Learning rate
    gamma = 0.6  # Discount factor
    
    # Create initial population
    population = np.random.randint(2, size=(population_size, n))
    
    for generation in range(generations):
        # Evaluate fitness
        fitness = np.dot(population, prize) - np.sum(population * weight, axis=1)
        
        # Selection
        sorted_indices = np.argsort(fitness)[::-1]
        population = population[sorted_indices][:elite_size]
        
        # Crossover
        offspring = np.zeros((population_size - elite_size, n))
        for i in range(0, population_size - elite_size, 2):
            parent1, parent2 = population[i], population[i+1]
            crossover_point = np.random.randint(1, n)
            offspring[i] = np.concatenate([parent1[:crossover_point], parent2[crossover_point:]])
            offspring[i+1] = np.concatenate([parent2[:crossover_point], parent1[crossover_point:]])
        
        population = np.concatenate([population, offspring])
        
        # Mutation
        for i in range(population_size):
            if np.random.rand() < mutation_rate:
                mutation_point = np.random.randint(n)
                population[i][mutation_point] = 1 - population[i][mutation_point]
        
        # Stochastic Local Search
        for _ in range(local_search_iterations):
            for i in range(population_size):
                for j in range(n):
                    if np.random.rand() < temperature:
                        population[i][j] = 1 - population[i][j]
                        # Update fitness
                        fitness[i] = np.dot(population[i], prize) - np.sum(population[i] * weight, axis=1)
        
        # Reinforcement Learning
        for i in range(population_size):
            for j in range(n):
                reward = np.dot(population[i], prize) - np.sum(population[i] * weight, axis=1)
                action_value = np.dot(population[i], prize[j]) - weight[j]
                heuristics[j] += alpha * (reward + gamma * max(heuristics) - heuristics[j])
    
    # Return the most promising items
    return np.where(heuristics > 0)[0]
