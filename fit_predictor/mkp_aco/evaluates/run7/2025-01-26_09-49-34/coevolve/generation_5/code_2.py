import numpy as np
import numpy as np

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    n, m = prize.shape[0], weight.shape[1]
    fitness = np.zeros(n)
    heuristic = np.zeros(n)
    max_prize = np.max(prize)
    min_weight = np.min(weight)
    max_weight = np.max(weight)
    
    # Adaptive Stochastic Sampling
    for i in range(n):
        # Dynamic fitness evaluation
        for j in range(m):
            if weight[i, j] < min_weight or weight[i, j] > max_weight:
                fitness[i] += prize[i] * (1 - weight[i, j] / max_weight)
            else:
                fitness[i] += prize[i] * (1 - weight[i, j] / min_weight)
        
        # Robust perturbation techniques
        if np.random.rand() < 0.1:  # 10% chance to apply perturbation
            if np.random.rand() < 0.5:
                perturbation = np.random.uniform(-0.1, 0.1)
                fitness[i] = max(0, min(fitness[i] + perturbation, max_prize))
            else:
                perturbation = np.random.uniform(0.1, 0.5)
                fitness[i] = min(max_prize, fitness[i] + perturbation)
        
        # Update heuristic based on dynamic fitness
        heuristic[i] = fitness[i] / max(fitness)
    
    return heuristic
