import numpy as np
import numpy as np

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    # Calculate value-to-weight ratio for each item
    value_to_weight_ratio = prize / weight.sum(axis=1)
    
    # Normalize the ratio to sum to 1 across all items
    normalized_ratio = value_to_weight_ratio / value_to_weight_ratio.sum()
    
    # Rank items based on normalized ratio
    ranking = np.argsort(-normalized_ratio)
    
    # Calculate the heuristics score for each item
    heuristics = np.zeros_like(prize)
    for i in ranking:
        # Assign higher scores to higher ranked items
        heuristics[i] = 1.0 / (i + 1)
    
    return heuristics