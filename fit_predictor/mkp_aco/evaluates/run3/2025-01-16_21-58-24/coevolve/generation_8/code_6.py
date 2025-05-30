import numpy as np
import numpy as np

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    n = prize.shape[0]
    m = weight.shape[1]
    
    # Calculate the heuristic value for each item based on the prize and weight
    # In this simple heuristic, we use the ratio of prize to the sum of weights in each dimension
    # This is just a placeholder heuristic and can be replaced with a more sophisticated one
    heuristic_values = np.zeros(n)
    for i in range(n):
        weight_sum = np.sum(weight[i])
        if weight_sum > 0:
            heuristic_values[i] = np.sum(prize[i]) / weight_sum
    
    return heuristic_values