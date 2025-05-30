import numpy as np
import numpy as np

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    # Initialize heuristics array with zeros
    heuristics = np.zeros_like(prize)
    
    # Calculate the weighted ratio index for each item
    for i in range(prize.shape[0]):
        # Sum of the weights for the current item
        total_weight = np.sum(weight[i])
        # If the total weight is not zero, calculate the ratio
        if total_weight != 0:
            heuristics[i] = np.sum(prize[i]) / total_weight
    
    # Apply adaptive probabilistic sampling to adjust heuristics
    # Here, we use a simple example where we multiply heuristics by a random factor
    random_factor = np.random.rand(prize.shape[0])
    heuristics *= random_factor
    
    # Normalize the heuristics to ensure they sum to 1
    heuristics /= np.sum(heuristics)
    
    return heuristics