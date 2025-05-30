import numpy as np
import numpy as np

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    n = prize.shape[0]
    m = weight.shape[1]
    heuristics = np.zeros(n)
    
    # Probabilistic sampling to explore diverse solution landscapes
    for i in range(n):
        # Calculate the weighted value of each item
        weighted_value = np.sum(prize[i] * weight[i])
        # Calculate the adherence to constraints (all constraints are 1 in this case)
        adherence = np.sum(weight[i])
        # Fitness function based on weighted value and adherence
        fitness = weighted_value / (adherence + 1e-6)  # Adding a small constant to avoid division by zero
        heuristics[i] = fitness
    
    return heuristics