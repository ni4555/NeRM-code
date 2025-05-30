import numpy as np
import numpy as np

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    # Normalize the value-to-weight ratio for each item
    normalized_value_to_weight = prize / weight
    
    # Calculate the average normalized value-to-weight ratio
    average_ratio = np.mean(normalized_value_to_weight)
    
    # Normalize the ratios to create a probabilistic heuristic
    normalized_probabilities = normalized_value_to_weight / average_ratio
    
    # Calculate the cumulative sum of probabilities in descending order
    cumulative_probabilities = np.cumsum(normalized_probabilities)[::-1]
    
    # Initialize an array to store the heuristics
    heuristics = np.zeros_like(normalized_probabilities)
    
    # Assign the cumulative probabilities as heuristics
    heuristics = cumulative_probabilities
    
    return heuristics