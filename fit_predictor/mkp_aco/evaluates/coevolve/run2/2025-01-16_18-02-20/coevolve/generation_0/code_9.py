import numpy as np
import numpy as np

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    # Calculate the prize to weight ratio for each item
    prize_to_weight_ratio = prize / weight
    
    # Normalize the ratios to get a score between 0 and 1
    max_ratio = np.max(prize_to_weight_ratio)
    heuristics = prize_to_weight_ratio / max_ratio
    
    return heuristics