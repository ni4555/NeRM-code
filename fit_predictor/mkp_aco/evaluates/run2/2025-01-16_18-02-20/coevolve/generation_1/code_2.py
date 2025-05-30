import numpy as np
import numpy as np

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    # Assuming that each item has a fixed weight constraint of 1 in each dimension,
    # the heuristic can be a simple ratio of prize to weight for each item.
    # However, without specific details on how to compute "promising", we will use
    # a placeholder heuristic. The actual heuristic should be designed based on
    # the problem's requirements and constraints.
    
    # Example heuristic: inverse of the sum of weights (assuming each weight dimension is 1)
    # This heuristic suggests that items with lower total weight are more promising.
    heuristic_values = 1.0 / (weight.sum(axis=1) + 1e-8)  # Adding a small constant to avoid division by zero
    
    # The resulting heuristic values are directly returned.
    return heuristic_values