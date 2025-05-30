import numpy as np
import numpy as np
from scipy.stats import poisson

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    # Normalize value-to-weight ratio for each item
    normalized_ratio = prize / weight
    
    # Calculate the sum of normalized ratios to use as a denominator for probabilities
    total_normalized_ratio = np.sum(normalized_ratio)
    
    # Normalize ratios to probabilities by dividing by the total sum
    probabilities = normalized_ratio / total_normalized_ratio
    
    # Use Poisson distribution to simulate the probabilistic selection of items
    # Poisson distribution with lambda equal to the probability of selecting an item
    heuristics = poisson.pmf(np.arange(1, len(probabilities) + 1), lambda=probabilities)
    
    # Normalize the heuristics to sum to 1 (probability distribution)
    heuristics /= np.sum(heuristics)
    
    return heuristics