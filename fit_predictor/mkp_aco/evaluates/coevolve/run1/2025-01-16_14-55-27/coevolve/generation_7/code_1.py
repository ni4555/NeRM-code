import numpy as np
import numpy as np

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    # Normalize the prize to a scale of 0 to 1
    prize_normalized = prize / np.sum(prize)
    
    # Calculate the weighted ratio index for each item
    weighted_ratio = prize_normalized * np.sum(weight, axis=1)
    
    # Normalize the weighted ratio to a scale of 0 to 1
    weighted_ratio_normalized = weighted_ratio / np.sum(weighted_ratio)
    
    # Integrate adaptive probabilistic sampling
    # For simplicity, we'll use a basic probabilistic sampling based on the normalized weighted ratio
    # Here, we could further integrate advanced normalization frameworks or other techniques
    
    # Calculate the heuristic for each item
    heuristics = weighted_ratio_normalized
    
    return heuristics