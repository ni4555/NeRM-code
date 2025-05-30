import numpy as np
import numpy as np

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    n, m = prize.shape[0], weight.shape[1]
    # Initialize a matrix to store the ratio of prize to weight for each item in each dimension
    prize_weight_ratio = np.zeros((n, m))
    for i in range(n):
        prize_weight_ratio[i] = prize[i] / weight[i]
    
    # Calculate the average ratio for each dimension
    avg_ratio = np.mean(prize_weight_ratio, axis=0)
    
    # Calculate the normalized ratio for each item in each dimension
    normalized_ratio = prize_weight_ratio / avg_ratio
    
    # Perform a stochastic sampling to select subsets of items
    random_indices = np.random.choice(n, size=n, replace=False)
    
    # Calculate the heuristics for each item based on normalized ratio
    heuristics = np.zeros(n)
    for i in range(n):
        heuristics[random_indices[i]] = np.sum(normalized_ratio[random_indices[i]])
    
    return heuristics
