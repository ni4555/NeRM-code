import numpy as np
import numpy as np

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    n, m = weight.shape
    # Assuming each dimension weight constraint is 1
    # Normalize weights by the maximum weight in each dimension
    weight_normalized = weight / np.sum(weight, axis=1, keepdims=True)
    # Calculate the heuristics based on the ratio of prize to normalized weight
    heuristics = prize / weight_normalized
    return heuristics