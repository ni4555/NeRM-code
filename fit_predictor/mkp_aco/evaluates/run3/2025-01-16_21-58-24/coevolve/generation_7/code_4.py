import numpy as np
import numpy as np

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    n, m = weight.shape
    # Initialize heuristic array with zeros
    heuristics = np.zeros(n)
    
    # Step 1: Probabilistic sampling to estimate initial heuristics
    for i in range(n):
        # Randomly select items and calculate the total weight
        random_indices = np.random.choice(n, size=m, replace=False)
        total_weight = np.sum(weight[random_indices])
        # Estimate the probability of including item i
        probability = np.exp(-total_weight)
        heuristics[i] = probability
    
    # Step 2: Adaptive Evolutionary Computation to refine heuristics
    population_size = 20
    elite_size = 5
    for _ in range(10):  # Number of evolutionary generations
        # Create a new population with random mutations based on the current heuristics
        new_population = np.random.normal(heuristics, 0.1, size=(population_size, n))
        new_population = np.clip(new_population, 0, 1)  # Keep probabilities between 0 and 1
        
        # Select the elite
        elite_indices = np.argsort(new_population, axis=1)[:, -elite_size:]
        elite = new_population[np.arange(population_size), elite_indices]
        
        # Replace the rest of the population with mutated individuals
        new_population[np.arange(population_size), :elite_size] = elite
        heuristics = new_population
    
    # Step 3: Robust Local Search to fine-tune the heuristics
    for i in range(n):
        for j in range(i + 1, n):
            # Swap the heuristics and evaluate the change
            heuristics[[i, j]] = heuristics[[j, i]]
            # Evaluate the new heuristics with a simple fitness function
            # Here we just calculate the total prize, assuming all constraints are satisfied
            total_prize = np.sum(prize * heuristics)
            # Swap back if the total prize is worse after the swap
            if total_prize < np.sum(prize * heuristics_v2(prize, weight)):
                heuristics[[i, j]] = heuristics[[j, i]]
    
    # Return the heuristics array
    return heuristics