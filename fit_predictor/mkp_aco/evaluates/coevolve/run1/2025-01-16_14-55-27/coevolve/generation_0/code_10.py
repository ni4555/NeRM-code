import numpy as np
import numpy as np

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    # Assuming the problem is to maximize the prize and the weight constraint for each dimension is 1,
    # we can create a heuristic based on the ratio of prize to weight for each item.
    # The heuristic for each item will be the maximum ratio for that item across all dimensions.
    # Since the weight constraint is 1 for each dimension, we can use the maximum prize value for the heuristic.
    
    # Create a heuristic based on the maximum prize for each item
    max_prize_per_item = np.max(prize, axis=1)
    
    # Calculate the heuristic as the maximum prize per item
    heuristics = max_prize_per_item
    
    return heuristics