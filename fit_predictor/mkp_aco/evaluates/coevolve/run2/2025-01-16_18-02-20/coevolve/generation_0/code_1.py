import numpy as np
import numpy as np

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    # Calculate the total prize for each item
    total_prize = np.sum(prize, axis=1)
    
    # Calculate the ratio of total prize to total weight for each item
    ratio = total_prize / np.sum(weight, axis=1)
    
    # Normalize the ratio by dividing with the maximum ratio
    normalized_ratio = ratio / np.max(ratio)
    
    # The normalized ratio is the heuristic value for each item
    heuristics = normalized_ratio
    
    return heuristics