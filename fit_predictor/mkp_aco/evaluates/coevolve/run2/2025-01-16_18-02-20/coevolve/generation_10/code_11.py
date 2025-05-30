import numpy as np
import numpy as np
from scipy.stats import beta

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    n = prize.shape[0]
    m = weight.shape[1]
    
    # Calculate value-to-weight ratio for each item
    value_to_weight_ratio = prize / weight.sum(axis=1)
    
    # Normalize the value-to-weight ratio to a probability distribution
    normalized_ratio = value_to_weight_ratio / value_to_weight_ratio.sum()
    
    # Use Beta distribution to sample probabilities
    # Alpha and beta parameters can be adjusted to control exploration vs exploitation
    alpha = 1
    beta_param = 1
    
    # Sample from the Beta distribution
    heuristics = beta.rvs(alpha + 1, beta_param + 1, size=n)
    
    # Normalize to ensure the sum of probabilities is 1
    heuristics /= heuristics.sum()
    
    return heuristics