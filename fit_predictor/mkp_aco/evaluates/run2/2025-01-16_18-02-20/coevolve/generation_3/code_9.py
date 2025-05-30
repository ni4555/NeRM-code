import numpy as np
import numpy as np

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    # Calculate the value-to-weight ratio for each item
    value_to_weight_ratio = prize / weight.sum(axis=1, keepdims=True)
    
    # Normalize the value-to-weight ratio to create a heuristic value
    max_ratio = np.max(value_to_weight_ratio)
    normalized_ratio = value_to_weight_ratio / max_ratio
    
    # Apply stochastic sampling with replacement
    random_indices = np.random.choice(len(value_to_weight_ratio), size=len(value_to_weight_ratio), replace=True)
    sampled_heuristics = normalized_ratio[random_indices]
    
    # Adjust the heuristic values to adapt to dynamic weight constraints
    # This is a simple example where we reduce the heuristic value if the weight is high
    dynamic_weight_factor = np.exp(-weight.sum(axis=1, keepdims=True) / np.mean(weight.sum(axis=1, keepdims=True)))
    adjusted_heuristics = sampled_heuristics * dynamic_weight_factor
    
    # Iterate over the items and select the most promising items based on the adjusted heuristics
    sorted_indices = np.argsort(adjusted_heuristics)[::-1]
    heuristics = np.zeros_like(value_to_weight_ratio)
    heuristics[sorted_indices] = adjusted_heuristics[sorted_indices]
    
    return heuristics