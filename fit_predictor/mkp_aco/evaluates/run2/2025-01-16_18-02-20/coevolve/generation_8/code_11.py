import numpy as np
import numpy as np
from scipy.stats import poisson

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    # Normalize the value-to-weight ratio for each item
    value_to_weight_ratio = prize / weight.sum(axis=1)
    
    # Calculate the maximum ratio to normalize all ratios to be between 0 and 1
    max_ratio = np.max(value_to_weight_ratio)
    normalized_ratio = value_to_weight_ratio / max_ratio
    
    # Define a probability distribution based on the normalized ratio
    # Higher normalized ratios correspond to higher probabilities of being selected
    probabilities = normalized_ratio ** 2  # Squaring to emphasize higher ratios
    
    # Normalize the probabilities to ensure they sum to 1
    probabilities /= probabilities.sum()
    
    # Generate heuristics based on the probabilities
    heuristics = np.random.choice(np.arange(len(prize)), size=len(prize), p=probabilities)
    
    return heuristics