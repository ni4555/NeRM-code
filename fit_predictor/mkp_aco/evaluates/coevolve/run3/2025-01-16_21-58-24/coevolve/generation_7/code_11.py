import numpy as np
import numpy as np

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    n, m = prize.shape[0], weight.shape[1]
    
    # Initialize a matrix to store the heuristic values for each item
    heuristic_matrix = np.zeros((n, n))
    
    # Probabilistic sampling to explore diverse solution landscapes
    for _ in range(100):  # Number of samples can be tuned
        # Sample a random subset of items
        random_subset_indices = np.random.choice(n, size=int(n * 0.5), replace=False)
        random_subset_prize = prize[random_subset_indices]
        random_subset_weight = weight[random_subset_indices]
        
        # Calculate the total prize for the random subset
        total_prize = np.sum(random_subset_prize)
        total_weight = np.sum(random_subset_weight, axis=1)
        
        # Calculate the heuristic values based on the ratio of prize to weight
        heuristic_matrix[random_subset_indices, :] = random_subset_prize / total_weight
    
    # Adaptive evolutionary computation to exploit promising regions
    # Initialize population with random solutions
    population_size = 20
    population = np.random.rand(population_size, n)
    
    for generation in range(50):  # Number of generations can be tuned
        # Evaluate fitness of each individual
        fitness = np.sum(prize * population, axis=1)
        
        # Select parents based on fitness
        parents_indices = np.argsort(fitness)[-population_size // 2:]
        parents = population[parents_indices]
        
        # Crossover and mutation to create new offspring
        offspring = np.random.choice(parents, size=population_size, replace=True)
        for i in range(population_size):
            # Perform crossover and mutation
            crossover_point = np.random.randint(1, n)
            offspring[i, :crossover_point] = parents[i, :crossover_point]
            offspring[i, crossover_point:] = parents[np.random.randint(population_size), crossover_point:]
            offspring[i] = np.where(np.random.rand(n) < 0.1, offspring[i], population[i])
        
        # Replace the old population with the new offspring
        population = offspring
    
    # Robust local search to refine solutions
    for item in range(n):
        # Calculate the current total prize excluding the current item
        current_total_prize = np.sum(prize) - prize[item]
        current_total_weight = np.sum(weight) - weight[item]
        
        # Calculate the heuristic value if the current item is included
        heuristic_value = prize[item] / current_total_weight
        
        # If the heuristic value is higher, update the heuristic matrix
        if heuristic_value > heuristic_matrix[item, item]:
            heuristic_matrix[item, item] = heuristic_value
    
    # Return the final heuristic values for each item
    return heuristic_matrix[:, 0]