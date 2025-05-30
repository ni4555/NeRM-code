import numpy as np
import numpy as np
from scipy.stats import multinomial

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    # Calculate the value-to-weight ratio for each item
    value_to_weight_ratio = prize / weight
    
    # Normalize the value-to-weight ratio to ensure all ratios are positive
    min_ratio = np.min(value_to_weight_ratio)
    value_to_weight_ratio = value_to_weight_ratio - min_ratio
    
    # Normalize the ratios to sum to 1
    normalized_ratio = value_to_weight_ratio / np.sum(value_to_weight_ratio)
    
    # Sample heuristics for each item using multinomial distribution
    heuristics = multinomial.pmf(np.arange(len(normalized_ratio)), n=1, p=normalized_ratio)
    
    return np.array(heuristics, dtype=float)