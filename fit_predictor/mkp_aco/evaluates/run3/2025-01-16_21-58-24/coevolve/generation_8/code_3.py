import numpy as np
import numpy as np

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    # Calculate the heuristic score for each item
    # Assuming the heuristic is a simple ratio of prize to weight for each dimension
    # Since all constraints are fixed to 1, we use the minimum weight across all dimensions for each item
    min_weight_per_item = np.min(weight, axis=1, keepdims=True)
    heuristic_scores = prize / min_weight_per_item
    
    # Normalize the heuristic scores to ensure they are all non-negative
    heuristic_scores[heuristic_scores < 0] = 0
    
    return heuristic_scores