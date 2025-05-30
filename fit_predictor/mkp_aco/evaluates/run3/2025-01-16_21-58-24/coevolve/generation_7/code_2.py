import numpy as np
import numpy as np

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    # Initialize the heuristic values to zero
    heuristics = np.zeros_like(prize)
    
    # Probabilistic sampling to explore diverse solution landscapes
    num_samples = 100
    for _ in range(num_samples):
        # Randomly sample a subset of items
        indices = np.random.choice(weight.shape[0], size=weight.shape[0], replace=False)
        sampled_prize = prize[indices]
        sampled_weight = weight[indices]
        
        # Calculate the probability of each item being included based on its prize and weight
        probabilities = sampled_prize / np.sum(sampled_prize)
        
        # Update the heuristic values based on the probability of each item
        heuristics[indices] += probabilities
    
    # Adaptive evolutionary computation to exploit promising regions
    num_generations = 10
    population_size = 50
    for _ in range(num_generations):
        # Create a new population based on the current heuristics
        population = np.random.choice(range(weight.shape[0]), size=population_size, p=heuristics/np.sum(heuristics))
        
        # Evaluate the fitness of each individual in the population
        fitness = np.sum(prize[population] - weight[population], axis=1)
        
        # Select the top individuals for the next generation
        new_population = np.argsort(fitness)[-population_size//2:]
        
        # Update the heuristics based on the new population
        heuristics[population] = fitness[new_population]
    
    # Robust local search algorithms to refine solutions
    for i in range(weight.shape[0]):
        # Local search to improve the heuristic value of the current item
        neighbors = np.random.choice(range(weight.shape[0]), size=5, replace=False)
        neighbor_fitness = np.sum(prize[neighbors] - weight[neighbors], axis=1)
        
        # Update the heuristic if a better neighbor is found
        if np.any(neighbor_fitness > heuristics[neighbors]):
            heuristics[neighbors] = neighbor_fitness
    
    return heuristics