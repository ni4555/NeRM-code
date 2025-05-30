import numpy as np
import numpy as np

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    # Calculate normalized value for each item
    normalized_value = prize / np.sum(weight, axis=1, keepdims=True)
    
    # Initialize an array to store heuristic scores
    heuristics = np.zeros_like(prize)
    
    # Iterate over each item and calculate its heuristic score
    for i in range(prize.shape[0]):
        # Calculate the normalized weight sum for the current item
        current_weight_sum = np.sum(weight[i] * weight)
        
        # Update the heuristic score based on normalized value and normalized weight sum
        heuristics[i] = normalized_value[i] / current_weight_sum
    
    return heuristics