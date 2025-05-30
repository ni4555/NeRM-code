import numpy as np
import numpy as np

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    # Initialize the heuristics array with zeros
    heuristics = np.zeros_like(prize)
    
    # Compute the heuristics based on the prize and weight
    # For simplicity, we can use the ratio of prize to the sum of weights in each dimension
    # This is just a placeholder heuristic, actual implementation can be more complex
    for i in range(weight.shape[0]):
        # Calculate the sum of weights for the current item
        weight_sum = np.sum(weight[i])
        # Avoid division by zero
        if weight_sum > 0:
            # Calculate the heuristic value
            heuristics[i] = prize[i] / weight_sum
    
    return heuristics