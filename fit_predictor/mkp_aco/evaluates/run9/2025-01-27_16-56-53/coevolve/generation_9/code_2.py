import numpy as np
import numpy as np

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    n, m = prize.shape[0], weight.shape[1]
    heuristic = np.zeros(n)
    
    # Calculate heuristic based on the weighted sum of prizes and weights
    for i in range(n):
        total_prize = 0
        total_weight = 0
        for j in range(m):
            total_prize += prize[i] * weight[i, j]
            total_weight += weight[i, j]
        # Normalize by weight to get a per-item heuristic
        heuristic[i] = total_prize / total_weight if total_weight != 0 else 0
    
    # Adjust heuristics to balance exploration and exploitation
    # Here, we can use a simple method like penalizing high weights
    for i in range(n):
        heuristic[i] = heuristic[i] / (1 + weight[i].sum())
    
    return heuristic
