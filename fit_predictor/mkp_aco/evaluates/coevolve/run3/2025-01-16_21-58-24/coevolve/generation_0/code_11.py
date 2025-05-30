import numpy as np
import numpy as np

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    # Assuming that the weight of each item is 1 for all dimensions, we can simplify the problem
    # by using the sum of weights across all dimensions for each item.
    total_weight = weight.sum(axis=1)
    
    # Calculate the "prominence" of each item as the ratio of its prize to its total weight.
    # This heuristic assumes that items with a higher prize-to-weight ratio are more promising.
    prominence = prize / total_weight
    
    # We can use the prominence values as our heuristic scores.
    return prominence