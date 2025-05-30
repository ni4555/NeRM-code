import numpy as np
import numpy as np

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    # Assuming each dimension weight constraint is fixed to 1
    # The heuristic function can be a simple normalized value of the prize
    # divided by the sum of weights in each dimension
    # This assumes that the prize is high for items with high weight and vice versa
    normalized_prizes = prize / weight.sum(axis=1)
    # Normalize by summing to ensure that the sum of heuristics is 1
    heuristics = normalized_prizes / normalized_prizes.sum()
    return heuristics