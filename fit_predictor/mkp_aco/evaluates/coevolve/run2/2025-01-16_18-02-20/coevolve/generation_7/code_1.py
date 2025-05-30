import numpy as np
import numpy as np
from scipy.stats import poisson

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    # Calculate the value-to-weight ratio for each item
    value_to_weight_ratio = prize / weight
    
    # Calculate the normalized value-to-weight ratio
    normalized_ratio = value_to_weight_ratio / np.sum(value_to_weight_ratio)
    
    # Calculate the heuristic values based on a Poisson distribution
    # with the mean equal to the normalized value-to-weight ratio
    heuristics = poisson.pmf(range(1, len(prize) + 1), np.mean(normalized_ratio))
    
    return heuristics