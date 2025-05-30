import numpy as np
import numpy as np

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    # Calculate the total weight for each item by summing across dimensions
    total_weight = np.sum(weight, axis=1)
    
    # Calculate the heuristic score for each item
    # The heuristic score is a ratio of the prize to the total weight
    # This encourages items with higher prize relative to weight to be more promising
    heuristic_scores = prize / total_weight
    
    # Normalize the heuristic scores to make them sum to 1 (softmax)
    max_score = np.max(heuristic_scores)
    normalized_scores = heuristic_scores / max_score
    
    # Return the normalized heuristic scores as the heuristics for each item
    return normalized_scores