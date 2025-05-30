import numpy as np
import numpy as np

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    # Calculate the value-to-weight ratio for each item
    value_to_weight_ratio = prize / weight.sum(axis=1)
    
    # Normalize the ratios so that they sum to 1
    normalized_ratio = value_to_weight_ratio / value_to_weight_ratio.sum()
    
    # Rank the items based on their normalized ratio
    ranking = np.argsort(normalized_ratio)[::-1]
    
    # Calculate the heuristics score for each item based on the ranking
    heuristics = np.zeros_like(prize)
    heuristics[ranking] = normalized_ratio[ranking]
    
    return heuristics