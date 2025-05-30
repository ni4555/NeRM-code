import numpy as np
import numpy as np
from scipy.stats import truncnorm

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    n = prize.shape[0]
    m = weight.shape[1]
    
    # Normalize the value-to-weight ratio for each item
    normalized_value = prize / weight.sum(axis=1)
    
    # Normalize the weights for each dimension
    normalized_weights = weight / weight.sum(axis=1, keepdims=True)
    
    # Calculate the normalized value-to-weight ratio for each item
    normalized_value_to_weight = normalized_value * normalized_weights
    
    # Sample items with probability proportional to their normalized value-to-weight ratio
    # Use a truncated normal distribution to ensure that the probabilities are bounded between 0 and 1
    # and are truncated to the range of the normalized value-to-weight ratio
    trunc_normal = truncnorm(a=(normalized_value_to_weight.min() - np.inf) / (normalized_value_to_weight.max() - np.inf),
                             b=(np.inf - normalized_value_to_weight.min()) / (np.inf - normalized_value_to_weight.min()),
                             loc=normalized_value_to_weight.mean(),
                             scale=normalized_value_to_weight.std())
    
    # Sample the heuristics using the truncated normal distribution
    heuristics = trunc_normal.rvs(size=n)
    
    return heuristics