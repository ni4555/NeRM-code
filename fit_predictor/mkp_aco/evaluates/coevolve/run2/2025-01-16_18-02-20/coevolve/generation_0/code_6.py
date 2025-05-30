import numpy as np
import numpy as np

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    # Initialize the heuristics array with zeros
    heuristics = np.zeros_like(prize)
    
    # Calculate the heuristic for each item
    for i in range(prize.shape[0]):
        # Assuming we are using a simple heuristic that calculates the ratio of prize to weight
        # This heuristic assumes that the weight is a 2D array where each item has one weight
        # and that the weight constraint for each dimension is fixed to 1.
        # Therefore, the total weight of an item is the sum of its weights across all dimensions.
        total_weight = weight[i].sum()
        heuristics[i] = prize[i] / total_weight if total_weight > 0 else 0
    
    return heuristics