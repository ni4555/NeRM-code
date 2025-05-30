import numpy as np
import numpy as np

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    # Assuming the following optimization strategy:
    # - Normalize the prize for each item to ensure they're comparable across dimensions.
    # - Use a simple heuristic that calculates the prize-to-weight ratio for each item.
    # - Normalize the heuristic values to make them comparable across items.
    
    # Normalize the prize by dividing by the sum of all prizes to get a per-item value.
    normalized_prize = prize / np.sum(prize)
    
    # Calculate the prize-to-weight ratio for each item.
    prize_to_weight_ratio = normalized_prize / np.sum(weight, axis=1)
    
    # Normalize the ratio to get the heuristic values.
    max_ratio = np.max(prize_to_weight_ratio)
    min_ratio = np.min(prize_to_weight_ratio)
    heuristic = (prize_to_weight_ratio - min_ratio) / (max_ratio - min_ratio)
    
    return heuristic