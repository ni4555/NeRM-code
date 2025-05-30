import numpy as np
import numpy as np

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    # Calculate the value-to-weight ratio for each item
    value_to_weight_ratio = prize / weight.sum(axis=1)
    
    # Normalize the value-to-weight ratio to ensure it is between 0 and 1
    max_ratio = np.max(value_to_weight_ratio)
    normalized_ratio = value_to_weight_ratio / max_ratio
    
    # Apply a stochastic sampling strategy based on the normalized ratio
    # Here we use a simple random sampling with replacement, but this can be replaced
    # with more sophisticated sampling strategies if needed.
    random_indices = np.random.choice(range(len(normalized_ratio)), size=len(normalized_ratio), replace=True)
    sorted_indices = np.argsort(normalized_ratio)[random_indices]
    
    # The sorted indices represent the order of item selection based on their normalized value-to-weight ratio
    heuristics = np.zeros_like(normalized_ratio)
    heuristics[sorted_indices] = 1
    
    return heuristics