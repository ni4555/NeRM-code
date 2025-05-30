import numpy as np
import numpy as np
import random

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    n, m = prize.shape[0], weight.shape[1]
    heuristic_scores = np.zeros(n)
    
    # Initialize population
    population_size = 100
    population = [np.random.choice(n) for _ in range(population_size)]
    
    # Adaptive sampling mechanism
    sampling_rate = 0.1
    for generation in range(100):
        # Evaluate fitness
        fitness_scores = np.dot(prize[population], weight[population, 0])
        
        # Select top individuals
        top_individuals = population[np.argsort(fitness_scores)[-int(population_size * 0.1):]]
        
        # Stochastic optimization algorithm
        for individual in population:
            if random.random() < sampling_rate:
                neighbor = population[np.random.randint(population_size)]
                if fitness_scores[neighbor] > fitness_scores[individual]:
                    population[individual] = neighbor
        
        # Reinforcement learning framework
        for individual in top_individuals:
            heuristic_scores[individual] += 1
    
    # Normalize heuristic scores
    heuristic_scores /= np.max(heuristic_scores)
    
    return heuristic_scores
