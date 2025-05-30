import numpy as np
import numpy as np
from scipy.stats import multinomial

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    # Calculate the value-to-weight ratio for each item
    value_to_weight_ratio = prize / weight
    
    # Normalize the ratios to sum to 1
    normalized_ratios = value_to_weight_ratio / np.sum(value_to_weight_ratio)
    
    # Generate a sample of heuristics using a multinomial distribution
    num_samples = len(prize)
    heuristics = multinomial.rvs(p=normalized_ratios, size=num_samples)
    
    return heuristics