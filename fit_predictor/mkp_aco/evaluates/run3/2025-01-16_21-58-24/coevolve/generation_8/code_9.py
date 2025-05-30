import numpy as np
import numpy as np

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    # Initialize the heuristics array with zeros
    heuristics = np.zeros_like(prize)
    
    # Calculate the heuristics based on the prize and weight
    # Here we use a simple heuristic that takes the ratio of prize to weight
    # and multiplies it by the sum of the weights in each dimension to account for the dimension constraint.
    for i in range(prize.shape[0]):
        item_value = prize[i]
        item_weight = weight[i]
        # Normalize by the sum of the weights to account for the dimension constraint
        normalized_weight = np.sum(item_weight)
        # Avoid division by zero
        if normalized_weight > 0:
            heuristics[i] = item_value / normalized_weight
    
    return heuristics