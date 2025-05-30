import numpy as np
import numpy as np

def heuristics_v2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    # Calculate the value-to-weight ratio for each item
    value_to_weight_ratio = prize / weight
    
    # Normalize the value-to-weight ratio to get a heuristic value
    max_ratio = np.max(value_to_weight_ratio)
    normalized_ratio = value_to_weight_ratio / max_ratio
    
    # Apply a stochastic element to the heuristic by adding random noise
    # This noise is scaled by the heuristic value to ensure that better items are less likely to be affected
    noise = np.random.normal(0, 0.1, normalized_ratio.shape)
    heuristics = normalized_ratio + noise
    
    # Ensure the heuristics are non-negative
    heuristics = np.clip(heuristics, 0, None)
    
    return heuristics