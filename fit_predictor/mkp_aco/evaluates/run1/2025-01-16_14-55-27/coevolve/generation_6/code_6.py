import numpy as np
import numpy as np

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    n = prize.shape[0]
    m = weight.shape[1]
    
    # Calculate the weighted ratio for each item in each dimension
    weighted_ratio = prize / weight
    
    # Calculate the average weighted ratio for all dimensions
    avg_weighted_ratio = np.mean(weighted_ratio, axis=1)
    
    # Calculate the heuristic score based on the average weighted ratio
    heuristics = avg_weighted_ratio / np.sum(avg_weighted_ratio)
    
    # Normalize the heuristic scores so that they sum to 1
    heuristics /= np.sum(heuristics)
    
    return heuristics