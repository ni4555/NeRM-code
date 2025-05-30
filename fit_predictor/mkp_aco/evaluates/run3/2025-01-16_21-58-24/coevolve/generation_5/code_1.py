import numpy as np
import numpy as np

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    # Calculate the fitness for each item based on the weighted combination of item values
    # and adherence to the multi-dimensional constraints
    # In this case, since each dimension has a constraint of 1, we use the max of the weights in each dimension
    # as the constraint adherence measure
    max_weight = np.max(weight, axis=1)
    # Calculate the normalized fitness by dividing the prize by the maximum weight in each dimension
    normalized_fitness = prize / max_weight
    # Return the normalized fitness as the heuristic value for each item
    return normalized_fitness