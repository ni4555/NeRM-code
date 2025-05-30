import numpy as np
import numpy as np

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    # Calculate value-to-weight ratio for each item
    value_to_weight_ratio = prize / weight.sum(axis=1)
    
    # Normalize the value-to-weight ratio to ensure items with the highest ratio are prioritized
    normalized_ratio = value_to_weight_ratio / value_to_weight_ratio.sum()
    
    # Apply stochastic sampling strategy to adapt to evolving weight constraints
    # Here we simply use the normalized ratio as a heuristic, in a real scenario this would be more complex
    heuristics = normalized_ratio
    
    return heuristics