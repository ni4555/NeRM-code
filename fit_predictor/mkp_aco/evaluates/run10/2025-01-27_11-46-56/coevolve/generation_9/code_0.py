import numpy as np
import numpy as np

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    # Calculate the sum of prizes for normalization
    total_prize = np.sum(prize)
    # Normalize the prizes and weights
    normalized_prize = prize / total_prize
    normalized_weight = weight / np.sum(weight, axis=1, keepdims=True)
    
    # Initialize heuristic scores
    heuristics = np.zeros_like(prize)
    
    # Calculate heuristic scores
    for i in range(prize.shape[0]):
        for j in range(weight.shape[1]):
            heuristics[i] += normalized_prize[i] * normalized_weight[i, j]
    
    return heuristics
