import numpy as np
import numpy as np

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    # Normalize the prize array to avoid dominance by high-value items
    prize_normalized = prize / np.sum(prize)
    
    # Calculate the value-to-weight ratio for each item
    value_to_weight_ratio = prize_normalized / np.sum(weight, axis=1)
    
    # Calculate the heuristic values based on the value-to-weight ratio
    heuristics = value_to_weight_ratio
    
    return heuristics