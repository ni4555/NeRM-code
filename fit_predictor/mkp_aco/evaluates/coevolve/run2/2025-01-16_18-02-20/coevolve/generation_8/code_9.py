import numpy as np
import numpy as np

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    # Calculate the value-to-weight ratio for each item
    value_to_weight_ratio = prize / weight
    
    # Normalize the value-to-weight ratio to avoid large number dominance
    max_ratio = np.max(value_to_weight_ratio)
    normalized_ratio = value_to_weight_ratio / max_ratio
    
    # Apply a probabilistic selection based on the normalized ratio
    # We use the exponential distribution for the probability function
    # This is a common approach in stochastic optimization
    exponential_sum = np.exp(normalized_ratio)
    probability = exponential_sum / np.sum(exponential_sum)
    
    # Generate a random number for each item and select if it falls below the cumulative probability
    cumulative_probability = np.cumsum(probability)
    random_numbers = np.random.rand(len(prize))
    heuristics = np.where(random_numbers < cumulative_probability, 1, 0)
    
    return heuristics