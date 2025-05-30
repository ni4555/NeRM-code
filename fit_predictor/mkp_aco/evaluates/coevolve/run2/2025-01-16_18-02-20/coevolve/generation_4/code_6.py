import numpy as np
import numpy as np

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    # Calculate the value-to-weight ratio for each item
    value_to_weight_ratio = prize / weight.sum(axis=1)
    
    # Normalize the ratio to get a probability distribution
    normalized_ratio = value_to_weight_ratio / value_to_weight_ratio.sum()
    
    # Sample the items based on the probability distribution
    random_indices = np.random.choice(len(prize), size=int(n * 0.1), replace=False, p=normalized_ratio)
    
    # Create the heuristics array with a higher value for sampled items
    heuristics = np.zeros(n)
    heuristics[random_indices] = 1
    
    return heuristics