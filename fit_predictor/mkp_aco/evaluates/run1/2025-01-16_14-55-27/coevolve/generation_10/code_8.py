import numpy as np
import numpy as np

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    # Calculate the value-to-weight ratio for each item
    value_to_weight_ratio = prize / weight.sum(axis=1)
    
    # Normalize the ratios to ensure a fair comparison across items with different weights
    max_ratio = np.max(value_to_weight_ratio)
    min_ratio = np.min(value_to_weight_ratio)
    normalized_ratio = (value_to_weight_ratio - min_ratio) / (max_ratio - min_ratio)
    
    # The heuristics are the normalized value-to-weight ratios
    heuristics = normalized_ratio
    return heuristics