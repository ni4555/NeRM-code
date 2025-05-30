import numpy as np
import numpy as np

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    n, m = prize.shape
    probabilities = np.zeros(n)
    
    # Adaptive sampling based on current cumulative prize and weight
    cumulative_prize = np.sum(prize)
    cumulative_weight = np.sum(weight, axis=1)
    for i in range(n):
        probabilities[i] = (prize[i] / cumulative_prize) * (1 - np.sum(cumulative_weight[i]))
    
    # Real-time fitness recalibration
    fitness = np.random.rand(n)
    for i in range(n):
        if weight[i].any() > 1:
            raise ValueError("Weights must sum to 1 for each item.")
        fitness[i] *= probabilities[i]
    
    # Resilient perturbation strategies
    perturbed_fitness = np.copy(fitness)
    perturbation_factors = np.random.normal(0, 0.1, n)
    for i in range(n):
        perturbed_fitness[i] = max(fitness[i] + perturbation_factors[i], 0)
    
    # Normalize the perturbed fitness
    normalized_fitness = perturbed_fitness / np.sum(perturbed_fitness)
    
    return normalized_fitness
