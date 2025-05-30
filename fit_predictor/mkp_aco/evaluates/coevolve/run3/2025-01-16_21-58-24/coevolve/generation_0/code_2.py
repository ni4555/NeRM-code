import numpy as np
import numpy as np

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    # Calculate the total weight for each item
    total_weight_per_item = np.sum(weight, axis=1)
    
    # Avoid division by zero if any item has zero weight
    total_weight_per_item[total_weight_per_item == 0] = 1
    
    # Calculate the prize-to-weight ratio for each item
    prize_to_weight_ratio = prize / total_weight_per_item
    
    # Normalize the prize-to-weight ratio so that the highest ratio gets the highest score
    max_ratio = np.max(prize_to_weight_ratio)
    heuristics = prize_to_weight_ratio / max_ratio
    
    return heuristics