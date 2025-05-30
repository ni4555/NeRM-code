import numpy as np
import numpy as np

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    # Calculate the value-to-weight ratio for each item
    value_to_weight_ratio = prize / weight.sum(axis=1)
    
    # Normalize the value-to-weight ratio to avoid bias towards heavier items
    normalized_ratio = value_to_weight_ratio / value_to_weight_ratio.sum()
    
    # Adjust the normalized ratio to make it suitable for probabilistic selection
    adjusted_normalized_ratio = normalized_ratio * (1 / np.sqrt(normalized_ratio))
    
    # Create a probability distribution for item selection
    probabilities = adjusted_normalized_ratio / adjusted_normalized_ratio.sum()
    
    # Sample items based on the probability distribution
    n = prize.size
    heuristics = np.random.choice([0, 1], size=n, p=probabilities)
    
    return heuristics