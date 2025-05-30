import numpy as np
import numpy as np
from scipy.stats import poisson

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    # Calculate the raw value-to-weight ratio for each item
    raw_ratio = prize / weight
    
    # Normalize the ratio to account for items with higher weight
    normalized_ratio = raw_ratio / (np.sum(raw_ratio) / n)
    
    # Calculate the probability of selecting each item based on the normalized ratio
    # The probability of selecting an item is the Poisson probability of selecting it at least once
    # in a Poisson distribution with parameter equal to the normalized ratio
    prob_select = 1 - poisson.cdf(normalized_ratio, 1)
    
    # Apply adaptive sampling by multiplying the probability with a factor that
    # depends on the normalized ratio to enhance the selection of higher ratio items
    adaptive_prob = prob_select * (1 + normalized_ratio)
    
    # Normalize the adaptive probabilities to sum to 1
    adaptive_prob /= np.sum(adaptive_prob)
    
    # Generate the heuristics array
    heuristics = adaptive_prob * n
    
    return heuristics