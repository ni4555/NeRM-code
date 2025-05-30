import numpy as np
import numpy as np

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    # Calculate the prominence score for each item
    # The prominence is calculated as the sum of the prize-to-weight ratios for each dimension
    prominence_scores = np.sum(prize / weight, axis=1)
    
    # Normalize the prominence scores to get the heuristics
    # We use the maximum prominence score as the normalization factor
    max_prominence = np.max(prominence_scores)
    heuristics = prominence_scores / max_prominence
    
    return heuristics