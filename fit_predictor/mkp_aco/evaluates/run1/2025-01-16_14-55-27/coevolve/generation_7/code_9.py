import numpy as np
import numpy as np

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    # Normalize prize to ensure a scale for comparison
    normalized_prize = prize / np.sum(prize)
    
    # Calculate the weighted ratio index
    weighted_ratio_index = np.dot(normalized_prize, weight)
    
    # Normalize the weighted ratio index to get a probability
    normalized_weighted_ratio_index = weighted_ratio_index / np.sum(weighted_ratio_index)
    
    # Calculate the heuristics score as the normalized weighted ratio index
    heuristics = normalized_weighted_ratio_index
    
    return heuristics