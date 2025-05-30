import numpy as np
import numpy as np

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    # Calculate the ratio of prize to weight for each item
    prize_to_weight_ratio = prize / weight
    
    # Integrate advanced ratio analysis to refine the heuristics
    # Here we use a simple heuristic where we prioritize items with higher ratios
    # and apply a penalty for items that are too heavy in any dimension.
    # This is a placeholder for more complex ratio analysis and heuristic algorithms.
    heuristic = prize_to_weight_ratio
    for w in weight:
        if w[0] > 1:  # If any dimension is greater than 1, apply a penalty
            heuristic *= 0.5
    
    return heuristic