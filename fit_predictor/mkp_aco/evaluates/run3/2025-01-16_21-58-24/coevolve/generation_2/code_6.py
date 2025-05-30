import numpy as np
import numpy as np

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    # Assuming that the heuristic function is a simple one that calculates
    # the prize per unit weight for each item and then normalizes this value
    # by the sum of all prize per unit weight to maintain the same scale.
    # This is a simplistic heuristic that assumes that higher prize per unit weight
    # is better, but this can be modified to fit the problem's specific requirements.
    
    # Calculate prize per unit weight for each item
    prize_per_unit_weight = prize / weight.sum(axis=1)
    
    # Normalize the prize per unit weight to get the heuristic
    heuristics = prize_per_unit_weight / prize_per_unit_weight.sum()
    
    return heuristics