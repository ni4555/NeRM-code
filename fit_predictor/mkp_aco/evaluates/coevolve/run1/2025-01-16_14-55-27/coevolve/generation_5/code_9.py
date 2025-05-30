import numpy as np
import numpy as np

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    # Initialize a heuristic array with zeros
    heuristics = np.zeros_like(prize)
    
    # Calculate the ratio of prize to weight for each item
    prize_to_weight_ratio = prize / weight
    
    # Normalize the ratio to get a probability distribution
    total_ratio = np.sum(prize_to_weight_ratio)
    normalized_ratio = prize_to_weight_ratio / total_ratio
    
    # Scale the normalized ratio to the range [0, 1]
    heuristics = normalized_ratio * 100
    
    # Sort the heuristics in descending order
    sorted_indices = np.argsort(heuristics)[::-1]
    
    # Return the sorted heuristics array
    return heuristics[sorted_indices]