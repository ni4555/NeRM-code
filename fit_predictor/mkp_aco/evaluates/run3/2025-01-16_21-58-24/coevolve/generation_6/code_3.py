import numpy as np
import numpy as np

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    # Ensure the weights are properly normalized
    # Since the weight dimension is m and the constraint is fixed to 1,
    # we'll sum the weights in each dimension and normalize
    weight_sum = np.sum(weight, axis=1)
    normalized_weight = weight / weight_sum[:, np.newaxis]

    # Calculate the heuristic value as the prize divided by the normalized weight
    heuristic = prize / normalized_weight
    
    return heuristic