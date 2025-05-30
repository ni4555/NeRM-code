import numpy as np
import numpy as np

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    # Normalize the prize and weight arrays to ensure all items have the same scale
    normalized_prize = prize / np.linalg.norm(prize)
    normalized_weight = weight / np.linalg.norm(weight, axis=1, keepdims=True)
    
    # Compute the heuristic value for each item based on the normalized prize and weight
    # The heuristic is a weighted combination of normalized prize and normalized weight
    # Each item's heuristic value is the sum of the product of its prize and weight in each dimension
    heuristics = np.sum(normalized_prize * normalized_weight, axis=1)
    
    return heuristics