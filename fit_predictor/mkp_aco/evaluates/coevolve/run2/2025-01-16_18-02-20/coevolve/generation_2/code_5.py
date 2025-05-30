import numpy as np
import numpy as np

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    # Normalize the prize values for each item
    normalized_prize = prize / np.sum(prize)
    
    # Initialize the heuristic values array
    heuristics = np.zeros_like(prize)
    
    # Calculate the heuristic by taking the ratio of normalized prize to weight
    heuristics = normalized_prize / weight
    
    # Adjust the heuristics based on the weight constraints (fixed to 1)
    heuristics = heuristics / np.sum(heuristics, axis=1, keepdims=True)
    
    return heuristics